// This file is a part of Julia. License is MIT: https://julialang.org/license

/*
  Generic Functions
  . method table and lookup
  . GF constructor
  . dispatch
  . static parameter inference
  . method specialization and caching, invoking type inference
*/
#include <stdlib.h>
#include <string.h>
#include "julia.h"
#include "julia_internal.h"
#ifndef _OS_WINDOWS_
#include <unistd.h>
#endif
#include "julia_assert.h"

// The compilation signature is not used to cache the method if the number of overlapping methods is greater than this
#define MAX_UNSPECIALIZED_CONFLICTS 32

#ifdef __cplusplus
extern "C" {
#endif

_Atomic(int) allow_new_worlds = 1;
JL_DLLEXPORT _Atomic(size_t) jl_world_counter = 1; // uses atomic acquire/release
jl_mutex_t world_counter_lock;
static _Atomic(size_t) jl_method_cache_insert_generation = 1;

static inline size_t jl_method_cache_insert_generation_load(void) JL_NOTSAFEPOINT
{
    return jl_atomic_load_acquire(&jl_method_cache_insert_generation);
}

static inline void jl_method_cache_inserted(void)
{
    jl_atomic_fetch_add(&jl_method_cache_insert_generation, 1);
}

JL_DLLEXPORT size_t jl_get_world_counter(void) JL_NOTSAFEPOINT
{
    jl_task_t *ct = jl_current_task;
    if (ct->ptls->in_pure_callback)
        return ~(size_t)0;
    return jl_atomic_load_acquire(&jl_world_counter);
}

JL_DLLEXPORT size_t jl_get_tls_world_age(void) JL_NOTSAFEPOINT
{
    return jl_current_task->world_age;
}

// Compute the maximum number of times to unroll Varargs{T}, based on
// m->max_varargs (if specified) or a heuristic based on the maximum number of
// non-varargs arguments for the function type of the method signature.
//
// If provided, `may_increase` is set to 1 if the returned value is
// heuristic-based and has a chance of increasing in the future.
static size_t get_max_varargs(
        jl_method_t *m,
        uint8_t *may_increase) JL_NOTSAFEPOINT
{
    size_t max_varargs = 1;
    if (may_increase != NULL)
        *may_increase = 0;

    if (m->max_varargs != UINT8_MAX) {
        max_varargs = m->max_varargs;
    }
    else {
        jl_datatype_t *dt1 = jl_nth_argument_datatype(m->sig, 1);
        jl_datatype_t *dt;
        if (jl_kwcall_type && dt1 == jl_kwcall_type)
            dt = jl_nth_argument_datatype(m->sig, 3);
        else
            dt = dt1;
        if (dt != NULL && !jl_is_type_type((jl_value_t*)dt) && dt != jl_kwcall_type) {
            if (may_increase != NULL)
                *may_increase = 1; // `max_args` can increase as new methods are inserted

            max_varargs = jl_atomic_load_relaxed(&dt->name->max_args) + 2;
            if (jl_kwcall_type && dt1 == jl_kwcall_type)
                max_varargs += 2;
            if (max_varargs > m->nargs)
                max_varargs -= m->nargs;
            else
                max_varargs = 0;
        }
    }
    return max_varargs;
}

/// ----- Handling for Julia callbacks ----- ///

JL_DLLEXPORT int8_t jl_is_in_pure_context(void)
{
    jl_task_t *ct = jl_current_task;
    return ct->ptls->in_pure_callback;
}

tracer_cb jl_newmeth_tracer = NULL;
JL_DLLEXPORT void jl_register_newmeth_tracer(void (*callback)(jl_method_t *tracee))
{
    jl_newmeth_tracer = (tracer_cb)callback;
}

void jl_call_tracer(tracer_cb callback, jl_value_t *tracee)
{
    jl_task_t *ct = jl_current_task;
    int last_in = ct->ptls->in_pure_callback;
    JL_TRY {
        ct->ptls->in_pure_callback = 1;
        callback(tracee);
        ct->ptls->in_pure_callback = last_in;
    }
    JL_CATCH {
        ct->ptls->in_pure_callback = last_in;
        jl_printf((JL_STREAM*)STDERR_FILENO, "WARNING: tracer callback function threw an error:\n");
        jl_static_show((JL_STREAM*)STDERR_FILENO, jl_current_exception(ct));
        jl_printf((JL_STREAM*)STDERR_FILENO, "\n");
        jl_fprint_backtrace(ios_safe_stderr);
    }
}

/// ----- Definitions for various internal TypeMaps ----- ///

static int8_t jl_cachearg_offset(void)
{
    return 0;
}

/// ----- Insertion logic for special entries ----- ///


uint_t speccache_hash(size_t idx, jl_value_t *data)
{
    jl_method_instance_t *ml = (jl_method_instance_t*)jl_svecref(data, idx); // This must always happen inside the lock
    jl_value_t *sig = ml->specTypes;
    if (jl_is_unionall(sig))
        sig = jl_unwrap_unionall(sig);
    return ((jl_datatype_t*)sig)->hash;
}

static int speccache_eq(size_t idx, const void *ty, jl_value_t *data, uint_t hv)
{
    if (idx >= jl_svec_len(data))
        return 0; // We got a OOB access, probably due to a data race
    jl_method_instance_t *ml = (jl_method_instance_t*)jl_svecref(data, idx);
    if (ml == NULL || (jl_value_t*)ml == jl_nothing)
        return 0; // slot not yet published, probably due to a data race
    jl_value_t *sig = ml->specTypes;
    if (ty == sig)
        return 1;
    uint_t h2 = ((jl_datatype_t*)(jl_is_unionall(sig) ? jl_unwrap_unionall(sig) : sig))->hash;
    if (h2 != hv)
        return 0;
    return jl_types_equal(sig, (jl_value_t*)ty);
}

// get or create the MethodInstance for a specialization
static jl_method_instance_t *jl_specializations_get_linfo_(jl_method_t *m JL_PROPAGATES_ROOT, jl_value_t *type, jl_svec_t *sparams, jl_method_instance_t *mi_insert)
{
    if (m->source == NULL && m->generator == NULL && jl_atomic_load_relaxed(&m->unspecialized) != NULL && m != jl_opaque_closure_method && !m->is_for_opaque_closure)
        return jl_atomic_load_relaxed(&m->unspecialized); // handle builtin methods
    jl_value_t *ut = jl_is_unionall(type) ? jl_unwrap_unionall(type) : type;
    JL_TYPECHK(specializations, datatype, ut);
    uint_t hv = ((jl_datatype_t*)ut)->hash;
    jl_genericmemory_t *speckeyset = NULL;
    jl_value_t *specializations = NULL;
    size_t i = -1, cl = 0, lastcl;
    for (int locked = 0; locked < 2; locked++) {
        if (locked) {
            if (!sparams) // can't insert without knowing this
                return NULL;
            JL_LOCK(&m->writelock);
        }
        lastcl = cl;
        speckeyset = jl_atomic_load_acquire(&m->speckeyset);
        specializations = jl_atomic_load_relaxed(&m->specializations);
        if (specializations == (jl_value_t*)jl_emptysvec)
            continue;
        if (!jl_is_svec(specializations)) {
            jl_method_instance_t *mi = (jl_method_instance_t*)specializations;
            if (jl_types_equal(mi->specTypes, type)) {
                if (locked)
                    JL_UNLOCK(&m->writelock);
                return mi;
            }
            continue;
        }
        cl = jl_svec_len(specializations);
        if (hv) {
            ssize_t idx = jl_smallintset_lookup(speckeyset, speccache_eq, type, specializations, hv, 0);
            if (idx != -1) {
                jl_method_instance_t *mi = (jl_method_instance_t*)jl_svecref(specializations, idx);
                if (locked)
                    JL_UNLOCK(&m->writelock);
                return mi;
            }
        }
        else {
            _Atomic(jl_method_instance_t*) *data = (_Atomic(jl_method_instance_t*)*)jl_svec_data(specializations);
            JL_GC_PUSH1(&specializations); // clang-sa doesn't realize this loop uses specializations
            // the last lastcl-i-1 elements are already checked when locked, so start search with the new elements only
            for (i += cl - lastcl; i > 0; i--) {
                jl_method_instance_t *mi = jl_atomic_load_relaxed(&data[i]);
                if ((jl_value_t*)mi == jl_nothing)
                    break;
                if (jl_types_equal(mi->specTypes, type)) {
                    if (locked)
                        JL_UNLOCK(&m->writelock);
                    JL_GC_POP();
                    return mi;
                }
            }
            // i points to the first unchecked element, or the place to insert
            JL_GC_POP();
        }
    }
    jl_method_instance_t *mi = mi_insert ? mi_insert : jl_get_specialized(m, type, sparams);
    if (specializations == (jl_value_t*)jl_emptysvec) {
        jl_atomic_store_release(&m->specializations, (jl_value_t*)mi);
        jl_gc_wb(m, mi);
    }
    else {
        JL_GC_PUSH1(&mi);
        if (!jl_is_svec(specializations)) {
            jl_method_instance_t *mi = (jl_method_instance_t*)specializations;
            jl_value_t *type = mi->specTypes;
            jl_value_t *ut = jl_is_unionall(type) ? jl_unwrap_unionall(type) : type;
            uint_t hv = ((jl_datatype_t*)ut)->hash;
            cl = 7;
            i = cl - 1;
            specializations = (jl_value_t*)jl_svec_fill(cl, jl_nothing);
            jl_svecset(specializations, hv ? 0 : i--, mi);
            jl_atomic_store_release(&m->specializations, specializations);
            jl_gc_wb(m, specializations);
            if (hv)
                jl_smallintset_insert(&m->speckeyset, (jl_value_t*)m, speccache_hash, 0, specializations);
        }
        if (hv) {
            _Atomic(jl_method_instance_t*) *data = (_Atomic(jl_method_instance_t*)*)jl_svec_data(specializations);
            for (i = 0; i < cl; i++) {
                jl_method_instance_t *mi = jl_atomic_load_relaxed(&data[i]);
                if ((jl_value_t*)mi == jl_nothing)
                    break;
                assert(!jl_types_equal(mi->specTypes, type));
            }
            // i points at the place to insert
        }
        if (hv ? (i + 1 >= cl || jl_svecref(specializations, i + 1) != jl_nothing) : (i <= 1 || jl_svecref(specializations, i - 2) != jl_nothing)) {
            size_t ncl = cl < 7 ? 7 : (cl*3)>>1;
            jl_svec_t *nc = jl_alloc_svec_uninit(ncl);
            if (i > 0)
                memcpy((char*)jl_svec_data(nc), jl_svec_data(specializations), sizeof(void*) * i);
            for (int j = 0; j < ncl - cl; j++)
                jl_svecset(nc, j+i, jl_nothing);
            if (i < cl)
                memcpy((char*)jl_svec_data(nc) + sizeof(void*) * (i + ncl - cl),
                       (char*)jl_svec_data(specializations) + sizeof(void*) * i,
                       sizeof(void*) * (cl - i));
            specializations = (jl_value_t*)nc;
            jl_atomic_store_release(&m->specializations, specializations);
            jl_gc_wb(m, specializations);
            if (!hv)
                i += ncl - cl;
        }
        assert(jl_svecref(specializations, i) == jl_nothing);
        jl_svecset(specializations, i, mi);
        if (hv)
            jl_smallintset_insert(&m->speckeyset, (jl_value_t*)m, speccache_hash, i, specializations);
        JL_GC_POP();
    }
    JL_UNLOCK(&m->writelock); // may gc
    return mi;
}

JL_DLLEXPORT jl_method_instance_t *jl_specializations_get_linfo(jl_method_t *m JL_PROPAGATES_ROOT, jl_value_t *type, jl_svec_t *sparams)
{
    return jl_specializations_get_linfo_(m, type, sparams, NULL);
}

jl_method_instance_t *jl_specializations_get_or_insert(jl_method_instance_t *mi)
{
    jl_method_t *m = mi->def.method;
    jl_value_t *type = mi->specTypes;
    jl_svec_t *sparams = mi->sparam_vals;
    return jl_specializations_get_linfo_(m, type, sparams, mi);
}

JL_DLLEXPORT jl_value_t *jl_specializations_lookup(jl_method_t *m, jl_value_t *type)
{
    jl_value_t *mi = (jl_value_t*)jl_specializations_get_linfo(m, type, NULL);
    if (mi == NULL)
        return jl_nothing;
    return mi;
}

JL_DLLEXPORT jl_value_t *jl_methtable_lookup(jl_value_t *type, size_t world)
{
    // TODO: this is sort of an odd lookup strategy (and the only user of
    // jl_typemap_assoc_by_type with subtype=0), while normally jl_gf_invoke_lookup would be
    // expected to be used instead
    struct jl_typemap_assoc search = {type, world, NULL};
    jl_typemap_entry_t *sf = jl_typemap_assoc_by_type(jl_atomic_load_relaxed(&jl_method_table->defs), &search, 0, /*subtype*/0);
    if (!sf)
        return jl_nothing;
    return sf->func.value;
}

// ----- MethodInstance specialization instantiation ----- //

jl_method_t *jl_mk_builtin_func(jl_datatype_t *dt, jl_sym_t *sname, jl_fptr_args_t fptr) JL_GC_DISABLED
{
    jl_value_t *params[2];
    params[0] = dt->name->wrapper;
    params[1] = jl_tparam0(jl_anytuple_type);
    jl_datatype_t *tuptyp = (jl_datatype_t*)jl_apply_tuple_type_v(params, 2);

    jl_typemap_entry_t *newentry = NULL;
    jl_method_t *m = NULL;
    JL_GC_PUSH3(&m, &newentry, &tuptyp);

    m = jl_new_method_uninit(jl_core_module);
    m->name = sname;
    m->module = jl_core_module;
    m->isva = 1;
    m->nargs = 2;
    jl_atomic_store_relaxed(&m->primary_world, 1);
    jl_atomic_store_relaxed(&m->dispatch_status, METHOD_SIG_LATEST_ONLY | METHOD_SIG_LATEST_WHICH);
    m->sig = (jl_value_t*)tuptyp;
    m->slot_syms = jl_an_empty_string;
    m->nospecialize = 0;
    m->nospecialize = ~m->nospecialize;

    jl_method_instance_t *mi = jl_get_specialized(m, (jl_value_t*)tuptyp, jl_emptysvec);
    jl_atomic_store_relaxed(&m->unspecialized, mi);
    jl_gc_wb(m, mi);

    jl_debuginfo_t *di = NULL;
    jl_svec_t *edges = jl_emptysvec;
    jl_code_instance_t *codeinst = jl_new_codeinst(mi, jl_nothing,
        (jl_value_t*)jl_any_type, (jl_value_t*)jl_any_type, jl_nothing, jl_nothing,
        0, 1, ~(size_t)0, 0, jl_nothing, di, edges);
    jl_atomic_store_relaxed(&codeinst->specptr.fptr1, fptr);
    jl_atomic_store_relaxed(&codeinst->invoke, jl_fptr_args);
    jl_mi_cache_insert(mi, codeinst);

    newentry = jl_typemap_alloc(tuptyp, NULL, jl_emptysvec,
            (jl_value_t*)m, 1, ~(size_t)0);
    jl_typemap_insert(&jl_method_table->defs, (jl_value_t*)jl_method_table, newentry, 0);

    newentry = jl_typemap_alloc(tuptyp, NULL, jl_emptysvec,
            (jl_value_t*)mi, 1, ~(size_t)0);
    jl_typemap_insert(&jl_method_table->cache->cache, (jl_value_t*)jl_method_table->cache, newentry, 0);

    JL_GC_POP();
    return m;
}

// only relevant for bootstrapping. otherwise fairly broken.
static int emit_codeinst_and_edges(jl_code_instance_t *codeinst)
{
    jl_value_t *code = jl_atomic_load_relaxed(&codeinst->inferred);
    if (code) {
        if (jl_atomic_load_relaxed(&codeinst->invoke) != NULL)
            return 1;
        if (code != jl_nothing) {
            JL_GC_PUSH1(&code);
            jl_method_instance_t *mi = jl_get_ci_mi(codeinst);
            jl_method_t *def = mi->def.method;
            if (jl_is_method(def))
                code = (jl_value_t*)jl_uncompress_ir(def, codeinst, (jl_value_t*)code);
            if (jl_is_code_info(code)) {
                jl_emit_codeinst_to_jit(codeinst, (jl_code_info_t*)code);
                if (0) {
                    // next emit all the invoke edges too (if this seems profitable)
                    jl_array_t *src = ((jl_code_info_t*)code)->code;
                    for (size_t i = 0; i < jl_array_dim0(src); i++) {
                        jl_value_t *stmt = jl_array_ptr_ref(src, i);
                        if (jl_is_expr(stmt) && ((jl_expr_t*)stmt)->head == jl_assign_sym)
                            stmt = jl_exprarg(stmt, 1);
                        if (jl_is_expr(stmt) && ((jl_expr_t*)stmt)->head == jl_invoke_sym) {
                            jl_value_t *invoke = jl_exprarg(stmt, 0);
                            if (jl_is_code_instance(invoke))
                                emit_codeinst_and_edges((jl_code_instance_t*)invoke);
                        }
                    }
                }
                JL_GC_POP();
                return 1;
            }
            JL_GC_POP();
        }
    }
    return 0;
}

// The policy of a method instance's code (compiler §2.3): its method's; a top-level thunk's is its
// module's, the setting's default for a neutral module.
static int mi_policy_uncached(jl_method_instance_t *mi) JL_NOTSAFEPOINT
{
    jl_value_t *def = mi->def.value;
    if (jl_is_method(def))
        return jl_method_policy((jl_method_t*)def);
    if (jl_is_module(def) && jl_options.trim == JL_TRIM_NO) {
        int8_t mf = ((jl_module_t*)def)->facts;
        if (mf == 1)
            return JL_POLICY_FACTS;
        if (mf == 2)
            return JL_POLICY_PLAIN;
    }
    return jl_compile_by_facts_all() ? JL_POLICY_FACTS : JL_POLICY_PLAIN;
}
// The same, cached in the method instance's flags at its first use: dispatch asks it at every call.
// A module's option set after one of its methods was first called does not change that method's
// policy (C-49).
STATIC_INLINE int mi_policy(jl_method_instance_t *mi) JL_NOTSAFEPOINT
{
    uint8_t c = (jl_atomic_load_relaxed(&mi->flags) & MI_FLAG_POLICY_MASK) >> MI_FLAG_POLICY_SHIFT;
    if (__likely(c != 0))
        return c - 1;
    int p = mi_policy_uncached(mi);
    jl_atomic_fetch_or_relaxed(&mi->flags, (uint8_t)((p + 1) << MI_FLAG_POLICY_SHIFT));
    return p;
}
// Whether the code of `mi` reached with `reach` is compiled by facts (compiler §2.3): its own
// policy, or for a method that takes the policy of the code reaching it, that code's; the runtime's
// own entries reach with the setting's default.
STATIC_INLINE int mi_policy_facts_at(jl_method_instance_t *mi, int reach) JL_NOTSAFEPOINT
{
    int p = mi_policy(mi);
    if (p != JL_POLICY_CALLER)
        return p;
    if (reach == JL_REACH_RUNTIME)
        return jl_compile_by_facts_all();
    return reach == JL_REACH_FACTS;
}
JL_DLLEXPORT int jl_mi_policy_at(jl_method_instance_t *mi, int reach) JL_NOTSAFEPOINT
{
    return mi_policy_facts_at(mi, reach);
}
// A CodeInstance of Julia's own owner records the policy it was compiled in; a lookup takes only
// one of the policy wanted.
STATIC_INLINE int ci_policy_is(jl_code_instance_t *ci, int facts) JL_NOTSAFEPOINT
{
    return ((jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_FACTS) != 0) == (facts != 0);
}
STATIC_INLINE int ci_policy_facts(jl_code_instance_t *ci) JL_NOTSAFEPOINT
{
    return (jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_FACTS) != 0;
}
// Records the policy a new CodeInstance is compiled in, before it is published.
JL_DLLEXPORT void jl_ci_set_policy(jl_code_instance_t *ci, int facts) JL_NOTSAFEPOINT
{
    if (ci->owner != jl_nothing)
        return;
    if (facts)
        jl_atomic_fetch_or_relaxed(&ci->flags, JL_CI_FLAGS_FACTS);
    else
        jl_atomic_fetch_and_relaxed(&ci->flags, (uint8_t)~JL_CI_FLAGS_FACTS);
}

// Opportunistic SOURCE_MODE_ABI cache lookup, only for bootstrapping.
static jl_code_instance_t *jl_method_inferred_with_abi(jl_method_instance_t *mi JL_PROPAGATES_ROOT, size_t world, int facts)
{
    jl_code_instance_t *codeinst = jl_atomic_load_relaxed(&mi->cache);
    for (; codeinst; codeinst = jl_atomic_load_relaxed(&codeinst->next)) {
        if (codeinst->owner != jl_nothing || !ci_policy_is(codeinst, facts))
            continue;
        if (jl_atomic_load_relaxed(&codeinst->min_world) <= world && world <= jl_atomic_load_relaxed(&codeinst->max_world)) {
            if (emit_codeinst_and_edges(codeinst))
                return codeinst;
        }
    }
    return NULL;
}

jl_mutex_t jl_typeinf_lock;

// run type inference on lambda "mi" for given argument types.
// returns the inferred source, and may cache the result in mi
// if successful, also updates the mi argument to describe the validity of this src
// if inference doesn't occur (or can't finish), returns NULL instead
jl_code_instance_t *jl_type_infer(jl_method_instance_t *mi, size_t world, uint8_t source_mode, uint8_t trim_mode)
{
    return jl_type_infer_policy(mi, world, source_mode, trim_mode, mi_policy_facts_at(mi, JL_REACH_RUNTIME));
}
// the same, for code of a policy (compiler §2.3)
jl_code_instance_t *jl_type_infer_policy(jl_method_instance_t *mi, size_t world, uint8_t source_mode, uint8_t trim_mode, int facts)
{
    if (jl_typeinf_func == NULL) {
        if (source_mode == SOURCE_MODE_ABI)
            return jl_method_inferred_with_abi(mi, world, facts);
        else
            return NULL;
    }
    jl_task_t *ct = jl_current_task;
    if (ct->reentrant_timing & 0b1000) {
        // We must avoid attempting to re-enter inference here
        assert(0 && "attempted to enter inference while writing out image");
        abort();
    }
    // In case we use higher bits later, mask them out
    if ((ct->reentrant_timing & 0b1111) >= 0b110)
        return NULL;

    jl_code_instance_t *ci = NULL;
#ifdef ENABLE_INFERENCE
    if (jl_engine_hasreserved_policy(mi, jl_nothing, facts)) // don't recur on a thread on the same MethodInstance--force it to interpret it until the inference has finished
        return NULL;
    JL_TIMING(INFERENCE, INFERENCE);
    jl_value_t **fargs;
    JL_GC_PUSH1(&ci);
    JL_GC_PUSHARGS(fargs, 6);
    fargs[0] = (jl_value_t*)jl_typeinf_func;
    fargs[1] = (jl_value_t*)mi;
    fargs[2] = jl_box_ulong(world);
    fargs[3] = jl_box_uint8(source_mode);
    fargs[4] = jl_box_uint8(trim_mode);
    fargs[5] = facts ? jl_true : jl_false;
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif

    jl_timing_show_method_instance(mi, JL_TIMING_DEFAULT_BLOCK);
#ifdef TRACE_INFERENCE
    if (mi->specTypes != (jl_value_t*)jl_emptytuple_type) {
        jl_printf(JL_STDERR,"inference on ");
        jl_static_show_func_sig(JL_STDERR, (jl_value_t*)mi->specTypes);
        jl_printf(JL_STDERR, "\n");
    }
#endif
    int last_pure = ct->ptls->in_pure_callback;
    ct->ptls->in_pure_callback = 0;
    size_t last_age = ct->world_age;
    ct->world_age = jl_typeinf_world;
    // the task world is frozen at jl_typeinf_world, strictly below every runtime
    // membership entry: ask memberships at the inference world instead (compiler §4.11)
    size_t last_mq = jl_set_membership_query_world(world);
    // first bit is for reentrant timing,
    // so adding 1 to the bit above performs
    // inference reentrancy counter addition.
    // Note that this is only safe because
    // the counter varies from 0-3; if we
    // increase that limit, we'll need to
    // allocate another bit for the counter.
    ct->reentrant_timing += 0b10;
    JL_LOCK(&jl_typeinf_lock);
    JL_TRY {
        ci = (jl_code_instance_t*)jl_apply(fargs, 6);
    }
    JL_CATCH {
        jl_value_t *e = jl_current_exception(ct);
        jl_printf((JL_STREAM*)STDERR_FILENO, "Internal error: during type inference of\n");
        jl_static_show_func_sig((JL_STREAM*)STDERR_FILENO, (jl_value_t*)mi->specTypes);
        jl_printf((JL_STREAM*)STDERR_FILENO, "\nEncountered ");
        if (e == jl_stackovf_exception) {
            jl_printf((JL_STREAM*)STDERR_FILENO, "stack overflow.\n");
            jl_printf((JL_STREAM*)STDERR_FILENO, "This might be caused by recursion over very long tuples or argument lists.\n");
        }
        else {
            jl_printf((JL_STREAM*)STDERR_FILENO, "unexpected error in runtime:\n");
            jl_static_show((JL_STREAM*)STDERR_FILENO, e);
            jl_printf((JL_STREAM*)STDERR_FILENO, "\n");
            jl_fprint_backtrace(ios_safe_stderr);
        }
        ci = NULL;
#ifndef JL_NDEBUG
        abort();
#endif
    }
    JL_UNLOCK(&jl_typeinf_lock);
    jl_set_membership_query_world(last_mq);
    ct->world_age = last_age;
    ct->reentrant_timing -= 0b10;
    ct->ptls->in_pure_callback = last_pure;
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;

    if (ci && !jl_is_code_instance(ci)) {
        ci = NULL;
    }

    // Record inference entrance backtrace if enabled
    if (ci) {
        jl_push_inference_entrance_backtraces((jl_value_t*)ci);
    }

    JL_GC_POP();
    JL_GC_POP();
#endif

    return ci;
}

// Attempt to run `Core.Compiler.code_typed` on the lambda "mi"
JL_DLLEXPORT jl_code_info_t *jl_gdbcodetyped1(jl_method_instance_t *mi, size_t world)
{
    jl_task_t *ct = jl_current_task;
    jl_code_info_t *ci = NULL;
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    int last_pure = ct->ptls->in_pure_callback;
    ct->ptls->in_pure_callback = 0;
    size_t last_age = ct->world_age;
    ct->world_age = jl_typeinf_world;
    size_t last_mq = jl_set_membership_query_world(world); // see jl_type_infer
    jl_value_t **fargs;
    JL_GC_PUSHARGS(fargs, 4);
    jl_module_t *CC = (jl_module_t*)jl_get_global_value(jl_core_module, jl_symbol("Compiler"), ct->world_age);
    if (CC != NULL && jl_is_module(CC)) {
        JL_GC_PROMISE_ROOTED(CC);
        fargs[0] = jl_get_global_value(CC, jl_symbol("NativeInterpreter"), ct->world_age);
        fargs[1] = jl_box_ulong(world);
        fargs[1] = jl_apply(fargs, 2);
        fargs[0] = jl_get_global_value(CC, jl_symbol("typeinf_code"), ct->world_age);
        fargs[2] = (jl_value_t*)mi;
        fargs[3] = jl_true;
        ci = (jl_code_info_t*)jl_apply(fargs, 4);
    }
    jl_set_membership_query_world(last_mq);
    ct->world_age = last_age;
    ct->ptls->in_pure_callback = last_pure;
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    if (ci && !jl_is_code_info(ci)) {
        ci = NULL;
    }
    JL_GC_POP();
    return ci;
}

JL_DLLEXPORT jl_value_t *jl_call_in_typeinf_world(jl_value_t **args, int nargs)
{
    jl_task_t *ct = jl_current_task;
    size_t last_age = ct->world_age;
    ct->world_age = jl_typeinf_world;
    // the freeze point of the Julia-driven inference path (Base.code_typed, every
    // Compiler._entry; compiler §4.11): membership questions during the run answer at the
    // caller's real world (last_age), not the frozen one; a query world already set by an
    // enclosing region (jl_type_infer, a declaration) wins
    size_t last_mq = jl_set_membership_query_world(jl_membership_query_world() ? jl_membership_query_world() : last_age);
    int last_pure = ct->ptls->in_pure_callback;
    ct->ptls->in_pure_callback = 0;
    jl_value_t *ret = jl_apply(args, nargs);
    ct->ptls->in_pure_callback = last_pure;
    ct->world_age = last_age;
    jl_set_membership_query_world(last_mq);
    return ret;
}

static jl_code_instance_t *jl_get_method_inferred_policy(
        jl_method_instance_t *mi JL_PROPAGATES_ROOT, jl_value_t *rettype,
        size_t min_world, size_t max_world, jl_debuginfo_t *di, jl_svec_t *edges, int facts);
JL_DLLEXPORT jl_code_instance_t *jl_get_method_inferred(
        jl_method_instance_t *mi JL_PROPAGATES_ROOT, jl_value_t *rettype,
        size_t min_world, size_t max_world, jl_debuginfo_t *di, jl_svec_t *edges)
{
    return jl_get_method_inferred_policy(mi, rettype, min_world, max_world, di, edges,
                                         mi_policy_facts_at(mi, JL_REACH_RUNTIME));
}
// the same, for code of a policy (compiler §2.3)
static jl_code_instance_t *jl_get_method_inferred_policy(
        jl_method_instance_t *mi JL_PROPAGATES_ROOT, jl_value_t *rettype,
        size_t min_world, size_t max_world, jl_debuginfo_t *di, jl_svec_t *edges, int facts)
{
    jl_value_t *owner = jl_nothing; // TODO: owner should be arg
    jl_code_instance_t *codeinst = jl_atomic_load_relaxed(&mi->cache);
    for (; codeinst; codeinst = jl_atomic_load_relaxed(&codeinst->next)) {
        if (jl_atomic_load_relaxed(&codeinst->min_world) == min_world &&
            jl_atomic_load_relaxed(&codeinst->max_world) == max_world &&
            jl_egal(codeinst->owner, owner) && ci_policy_is(codeinst, facts) &&
            jl_egal(codeinst->rettype, rettype)) {
            if (di == NULL)
                return codeinst;
            jl_debuginfo_t *debuginfo = jl_atomic_load_relaxed(&codeinst->debuginfo);
            if (di != debuginfo) {
                if (!(debuginfo == NULL && jl_atomic_cmpswap_relaxed(&codeinst->debuginfo, &debuginfo, di)))
                    if (!(debuginfo && jl_egal((jl_value_t*)debuginfo, (jl_value_t*)di)))
                        continue;
            }
            // TODO: this is implied by the matching worlds, since it is intrinsic, so do we really need to verify it?
            jl_svec_t *e = jl_atomic_load_relaxed(&codeinst->edges);
            if (e && jl_egal((jl_value_t*)e, (jl_value_t*)edges))
                return codeinst;
        }
    }
    codeinst = jl_new_codeinst(
        mi, owner, rettype, (jl_value_t*)jl_any_type, NULL, NULL,
        0, min_world, max_world, 0, jl_nothing, di, edges);
    jl_ci_set_policy(codeinst, facts);
    jl_mi_cache_insert(mi, codeinst);
    return codeinst;
}

JL_DLLEXPORT int jl_mi_cache_has_ci(jl_method_instance_t *mi,
                                    jl_code_instance_t *ci)
{
    jl_code_instance_t *codeinst = jl_atomic_load_relaxed(&mi->cache);
    while (codeinst) {
        if (codeinst == ci)
            return 1;
        codeinst = jl_atomic_load_relaxed(&codeinst->next);
    }
    return 0;
}

// look for something with an egal ABI and properties that is already in the JIT for a whole edge (target_world=0) or can be added to the JIT with new source just for target_world.
JL_DLLEXPORT jl_code_instance_t *jl_get_ci_equiv(jl_code_instance_t *ci JL_PROPAGATES_ROOT, size_t target_world) JL_NOTSAFEPOINT
{
    jl_value_t *def = ci->def;
    jl_method_instance_t *mi = jl_get_ci_mi(ci);
    jl_value_t *owner = ci->owner;
    jl_value_t *rettype = ci->rettype;
    size_t min_world = jl_atomic_load_relaxed(&ci->min_world);
    size_t max_world = jl_atomic_load_relaxed(&ci->max_world);
    jl_code_instance_t *codeinst = jl_atomic_load_relaxed(&mi->cache);
    while (codeinst) {
        if (codeinst != ci &&
            jl_atomic_load_relaxed(&codeinst->inferred) != NULL &&
            (target_world ? 1 : jl_atomic_load_relaxed(&codeinst->invoke) != NULL) &&
            jl_atomic_load_relaxed(&codeinst->min_world) <= (target_world ? target_world : min_world) &&
            jl_atomic_load_relaxed(&codeinst->max_world) >= (target_world ? target_world : max_world) &&
            jl_egal(codeinst->def, def) &&
            jl_egal(codeinst->owner, owner) &&
            (owner != jl_nothing || ci_policy_is(codeinst, ci_policy_facts(ci))) &&
            jl_egal(codeinst->rettype, rettype)) {
            return codeinst;
        }
        codeinst = jl_atomic_load_relaxed(&codeinst->next);
    }
    return ci;
}


JL_DLLEXPORT jl_code_instance_t *jl_new_codeinst(
        jl_method_instance_t *mi, jl_value_t *owner,
        jl_value_t *rettype, jl_value_t *exctype,
        jl_value_t *inferred_const, jl_value_t *inferred,
        int32_t const_flags, size_t min_world, size_t max_world,
        uint32_t effects, jl_value_t *analysis_results,
        jl_debuginfo_t *di, jl_svec_t *edges /*, int absolute_max*/)
{
    assert(min_world <= max_world && "attempting to set invalid world constraints");
    //assert((!jl_is_method(mi->def.value) || max_world != ~(size_t)0 || min_world <= 1 || edges == NULL || jl_svec_len(edges) != 0) && "missing edges");
    jl_task_t *ct = jl_current_task;
    jl_code_instance_t *codeinst = (jl_code_instance_t*)jl_gc_alloc(ct->ptls, sizeof(jl_code_instance_t),
            jl_code_instance_type);
    codeinst->def = (jl_value_t*)mi;
    codeinst->owner = owner;
    jl_atomic_store_relaxed(&codeinst->edges, edges);
    jl_atomic_store_relaxed(&codeinst->min_world, min_world);
    jl_atomic_store_relaxed(&codeinst->max_world, max_world);
    codeinst->rettype = rettype;
    codeinst->exctype = exctype;
    jl_atomic_store_release(&codeinst->inferred, inferred);
    if ((const_flags & 2) == 0)
        inferred_const = NULL;
    codeinst->rettype_const = inferred_const;
    jl_atomic_store_relaxed(&codeinst->debuginfo, (jl_value_t*)di == jl_nothing ? NULL : di);
    jl_atomic_store_relaxed(&codeinst->specptr.fptr, NULL);
    jl_atomic_store_relaxed(&codeinst->invoke, NULL);
    if ((const_flags & 1) != 0) {
        assert(const_flags & 2);
        jl_atomic_store_relaxed(&codeinst->invoke, jl_fptr_const_return);
    }
    codeinst->time_infer_total = 0;
    codeinst->time_infer_self = 0;
    jl_atomic_store_relaxed(&codeinst->time_compile, 0);
    // Julia's own owner records the policy the code is compiled in (§2.3): here the policy its
    // method instance has where the runtime reaches it; a caller that compiles it for code of the
    // other policy records that (jl_ci_set_policy)
    jl_atomic_store_relaxed(&codeinst->flags, owner == jl_nothing && mi_policy_facts_at(mi, JL_REACH_RUNTIME) ? JL_CI_FLAGS_FACTS : 0);
    jl_atomic_store_relaxed(&codeinst->precompile, 0);
    jl_atomic_store_relaxed(&codeinst->next, NULL);
    jl_atomic_store_relaxed(&codeinst->ipo_purity_bits, effects);
    codeinst->analysis_results = analysis_results;
    return codeinst;
}

JL_DLLEXPORT void jl_update_codeinst(
        jl_code_instance_t *codeinst, jl_value_t *inferred,
        int32_t const_flags, size_t min_world, size_t max_world,
        uint32_t effects, jl_value_t *analysis_results,
        double time_infer_total, double time_infer_cache_saved, double time_infer_self,
        jl_debuginfo_t *di, jl_svec_t *edges /* , int absolute_max*/)
{
    assert(min_world <= max_world && "attempting to set invalid world constraints");
    //assert((!jl_is_method(codeinst->def->def.value) || max_world != ~(size_t)0 || min_world <= 1 || jl_svec_len(edges) != 0) && "missing edges");
    codeinst->analysis_results = analysis_results;
    jl_gc_wb(codeinst, analysis_results);
    codeinst->time_infer_total = julia_double_to_half(time_infer_total);
    codeinst->time_infer_cache_saved = julia_double_to_half(time_infer_cache_saved);
    codeinst->time_infer_self = julia_double_to_half(time_infer_self);
    jl_atomic_store_relaxed(&codeinst->ipo_purity_bits, effects);
    jl_atomic_store_relaxed(&codeinst->debuginfo, di);
    jl_gc_wb(codeinst, di);
    jl_atomic_store_relaxed(&codeinst->edges, edges);
    jl_gc_wb(codeinst, edges);
    if ((const_flags & 1) != 0) {
        assert(codeinst->rettype_const);
        jl_atomic_store_release(&codeinst->invoke, jl_fptr_const_return);
    }
    jl_atomic_store_release(&codeinst->inferred, inferred);
    jl_gc_wb(codeinst, inferred);
    jl_atomic_store_relaxed(&codeinst->min_world, min_world); // XXX: these should be unchanged?
    jl_atomic_store_relaxed(&codeinst->max_world, max_world); // since the edges shouldn't change after jl_fill_codeinst
}

JL_DLLEXPORT void jl_fill_codeinst(
        jl_code_instance_t *codeinst,
        jl_value_t *rettype, jl_value_t *exctype,
        jl_value_t *inferred_const,
        int32_t const_flags, size_t min_world, size_t max_world,
        uint32_t effects, jl_value_t *analysis_results,
        jl_debuginfo_t *di, jl_svec_t *edges /* , int absolute_max*/)
{
    assert(min_world <= max_world && "attempting to set invalid world constraints");
    //assert((!jl_is_method(codeinst->def->def.value) || max_world != ~(size_t)0 || min_world <= 1 || jl_svec_len(edges) != 0) && "missing edges");
    codeinst->rettype = rettype;
    jl_gc_wb(codeinst, rettype);
    codeinst->exctype = exctype;
    jl_gc_wb(codeinst, exctype);
    if ((const_flags & 2) != 0) {
        codeinst->rettype_const = inferred_const;
        jl_gc_wb(codeinst, inferred_const);
    }
    jl_atomic_store_relaxed(&codeinst->edges, edges);
    jl_gc_wb(codeinst, edges);
    if ((jl_value_t*)di != jl_nothing) {
        jl_atomic_store_relaxed(&codeinst->debuginfo, di);
        jl_gc_wb(codeinst, di);
    }
    if ((const_flags & 1) != 0) {
        // TODO: may want to follow ordering restrictions here (see jitlayers.cpp)
        assert(const_flags & 2);
        jl_atomic_store_release(&codeinst->invoke, jl_fptr_const_return);
    }
    jl_atomic_store_relaxed(&codeinst->ipo_purity_bits, effects);
    codeinst->analysis_results = analysis_results;
    assert(jl_atomic_load_relaxed(&codeinst->min_world) == 1);
    assert(jl_atomic_load_relaxed(&codeinst->max_world) == 0);
    jl_atomic_store_release(&codeinst->inferred, jl_nothing);
    jl_atomic_store_release(&codeinst->min_world, min_world);
    jl_atomic_store_release(&codeinst->max_world, max_world);
}

JL_DLLEXPORT jl_code_instance_t *jl_new_codeinst_uninit(jl_method_instance_t *mi, jl_value_t *owner)
{
    jl_code_instance_t *codeinst = jl_new_codeinst(mi, owner, NULL, NULL, NULL, NULL, 0, 0, 0, 0, NULL, NULL, NULL);
    jl_atomic_store_relaxed(&codeinst->min_world, 1); // make temporarily invalid before returning, so that jl_fill_codeinst is valid later
    return codeinst;
}

JL_DLLEXPORT void jl_mi_cache_insert(jl_method_instance_t *mi JL_ROOTING_ARGUMENT,
                                     jl_code_instance_t *ci JL_ROOTED_ARGUMENT JL_MAYBE_UNROOTED)
{
    JL_GC_PUSH1(&ci);
    if (jl_is_method(mi->def.method))
        JL_LOCK(&mi->def.method->writelock);
    // Set native_cache_valid bit when inserting into cache
    jl_atomic_fetch_or_relaxed(&ci->flags, JL_CI_FLAGS_NATIVE_CACHE_VALID);
    // find the preferred location for insertion of ci now:
    //   - invoke+inferred group
    //   - inferred group
    //   - others group
    //   - unmoved
    //   - after existing entries with same applicable range
    jl_value_t *parent = (jl_value_t*)mi;
    _Atomic(jl_code_instance_t*) *slot = &mi->cache;
    jl_code_instance_t *oldci = jl_atomic_load_relaxed(slot);
    int hasinvoke = jl_atomic_load_relaxed(&ci->invoke) != NULL;
    int hasinferred = jl_atomic_load_relaxed(&ci->inferred) != NULL;
    size_t max_world = jl_atomic_load_relaxed(&ci->max_world);
    jl_code_instance_t *next = jl_atomic_load_relaxed(&ci->next);
    while (oldci) {
        if (oldci == ci)
            break;
        int old_hasinvoke = jl_atomic_load_relaxed(&oldci->invoke) != NULL;
        int old_hasinferred = jl_atomic_load_relaxed(&oldci->inferred) != NULL;
        size_t old_max_world = jl_atomic_load_relaxed(&oldci->max_world);
        if (hasinvoke && !old_hasinvoke)
            break;
        if (hasinferred && !old_hasinferred)
            break;
        if (next == NULL && old_max_world < max_world)
            break;
        parent = (jl_value_t*)oldci;
        slot = &oldci->next;
        oldci = jl_atomic_load_relaxed(slot);
    }
    if (oldci != ci) {
        jl_atomic_store_relaxed(&ci->next, oldci);
        if (oldci)
            jl_gc_wb(ci, oldci);
        jl_atomic_store_release(slot, ci);
        jl_gc_wb(parent, ci);
        if (oldci != NULL) {
            // list is now potentially circular, need to go find old pointer to ci starting from oldci and insert next there
            do {
                parent = (jl_value_t*)oldci;
                slot = &oldci->next;
                oldci = jl_atomic_load_relaxed(slot);
            } while (oldci && oldci != ci);
            if (oldci) {
                jl_atomic_store_release(slot, next);
                if (next)
                    jl_gc_wb(parent, next);
            }
        }
    }
    if (jl_is_method(mi->def.method))
        JL_UNLOCK(&mi->def.method->writelock);
    JL_GC_POP();
    return;
}

JL_DLLEXPORT int jl_mi_try_insert(jl_method_instance_t *mi JL_ROOTING_ARGUMENT,
                                   jl_code_instance_t *expected_ci,
                                   jl_code_instance_t *ci JL_ROOTED_ARGUMENT JL_MAYBE_UNROOTED)
{
    JL_GC_PUSH1(&ci);
    if (jl_is_method(mi->def.method))
        JL_LOCK(&mi->def.method->writelock);
    jl_code_instance_t *oldci = jl_atomic_load_relaxed(&mi->cache);
    int ret = 0;
    if (oldci == expected_ci) {
        jl_atomic_store_relaxed(&ci->next, oldci);
        if (oldci)
            jl_gc_wb(ci, oldci);
        jl_atomic_store_release(&mi->cache, ci);
        jl_gc_wb(mi, ci);
        ret = 1;
    }
    if (jl_is_method(mi->def.method))
        JL_UNLOCK(&mi->def.method->writelock);
    JL_GC_POP();
    return ret;
}

enum top_typename_facts {
    EXACTLY_ANY = 1 << 0,
    HAVE_TYPE = 1 << 1,
    EXACTLY_TYPE = 1 << 2,
    HAVE_FUNCTION = 1 << 3,
    EXACTLY_FUNCTION = 1 << 4,
    HAVE_KWCALL = 1 << 5,
    EXACTLY_KWCALL = 1 << 6,
    SHORT_TUPLE = 1 << 7,
};

static void foreach_top_nth_typename(void (*f)(jl_typename_t*, int, void*), jl_value_t *a JL_PROPAGATES_ROOT, int n, unsigned *facts, void *env)
{
    arraylist_t workqueue;
    arraylist_new(&workqueue, 0);

    // Push initial work item as jl_value_t* then int (cast to void*)
    arraylist_push(&workqueue, a);
    arraylist_push(&workqueue, (void*)(uintptr_t)n);

    while (workqueue.len > 0) {
        // Pop int n then jl_value_t* a (reverse order)
        int current_n = (int)(uintptr_t)arraylist_pop(&workqueue);
        jl_value_t *current_a = (jl_value_t*)arraylist_pop(&workqueue);
        JL_GC_PROMISE_ROOTED(current_a);

        if (jl_is_datatype(current_a)) {
            if (current_n <= 0) {
                jl_datatype_t *dt = ((jl_datatype_t*)current_a);
                if (dt->name == jl_type_typename) { // key Type{T} on T instead of Type
                    *facts |= HAVE_TYPE;
                    arraylist_push(&workqueue, jl_tparam0(current_a));
                    arraylist_push(&workqueue, (void*)(uintptr_t)-1);
                }
                else if (dt == jl_function_type) {
                    if (current_n == -1) // key Type{>:Function} as Type instead of Function
                        *facts |= EXACTLY_TYPE; // HAVE_TYPE is already set
                    else
                        *facts |= HAVE_FUNCTION | EXACTLY_FUNCTION;
                }
                else if (dt == jl_any_type) {
                    if (current_n == -1) // key Type{>:Any} and kinds as Type instead of Any
                        *facts |= EXACTLY_TYPE; // HAVE_TYPE is already set
                    else
                        *facts |= EXACTLY_ANY;
                }
                else if (dt == jl_kwcall_type) {
                    if (current_n == -1) // key Type{>:typeof(kwcall)} as exactly kwcall
                        *facts |= EXACTLY_KWCALL;
                    else
                        *facts |= HAVE_KWCALL;
                }
                else {
                    while (1) {
                        jl_datatype_t *super = dt->super;
                        if (super == jl_function_type) {
                            *facts |= HAVE_FUNCTION;
                            break;
                        }
                        if (super == jl_any_type || super->super == dt)
                            break;
                        dt = super;
                    }
                    f(dt->name, 1, env);
                }
            }
            else if (jl_is_tuple_type(current_a)) {
                if (jl_nparams(current_a) >= current_n) {
                    arraylist_push(&workqueue, jl_tparam(current_a, current_n - 1));
                    arraylist_push(&workqueue, (void*)(uintptr_t)0);
                }
                else
                    *facts |= SHORT_TUPLE;
            }
        }
        else if (jl_is_typevar(current_a)) {
            arraylist_push(&workqueue, ((jl_tvar_t*)current_a)->ub);
            arraylist_push(&workqueue, (void*)(uintptr_t)current_n);
        }
        else if (jl_is_unionall(current_a)) {
            arraylist_push(&workqueue, ((jl_unionall_t*)current_a)->body);
            arraylist_push(&workqueue, (void*)(uintptr_t)current_n);
        }
        else if (jl_is_uniontype(current_a)) {
            jl_uniontype_t *u = (jl_uniontype_t*)current_a;
            // Add both union branches to workqueue (push a second to visit first)
            arraylist_push(&workqueue, u->b);
            arraylist_push(&workqueue, (void*)(uintptr_t)current_n);
            arraylist_push(&workqueue, u->a);
            arraylist_push(&workqueue, (void*)(uintptr_t)current_n);
        }
    }

    arraylist_free(&workqueue);
}

// Inspect type `argtypes` for all backedge keys that might be relevant to it, splitting it
// up on some commonly observed patterns to make a better distribution.
// (It could do some of that balancing automatically, but for now just hard-codes kwcall.)
// Along the way, record some facts about what was encountered, so that those additional
// calls can be added later if needed for completeness.
// The `int explct` argument instructs the caller if the callback is due to an exactly
// encountered type or if it rather encountered a subtype.
// This is not capable of walking to all top-typenames for an explicitly encountered
// Function or Any, so the caller a fallback that can scan the entire  in that case.
// We do not de-duplicate calls when encountering a Union.
static int jl_foreach_top_typename_for(void (*f)(jl_typename_t*, int, void*), jl_value_t *argtypes JL_PROPAGATES_ROOT, int all_subtypes, void *env)
{
    unsigned facts = 0;
    foreach_top_nth_typename(f, argtypes, 1, &facts, env);
    if (facts & HAVE_KWCALL) {
        // split kwcall on the 3rd argument instead, using the same logic
        unsigned kwfacts = 0;
        foreach_top_nth_typename(f, argtypes, 3, &kwfacts, env);
        // copy kwfacts to original facts
        if (kwfacts & SHORT_TUPLE)
            kwfacts |= (all_subtypes ? EXACTLY_ANY : EXACTLY_KWCALL);
        facts |= kwfacts;
    }
    if (all_subtypes && (facts & (EXACTLY_FUNCTION | EXACTLY_TYPE | EXACTLY_ANY)))
        // flag that we have an explct match than is necessitating a full table scan
        return 0;
    // or inform caller of only which supertypes are applicable
    if (facts & HAVE_FUNCTION)
        f(jl_function_type->name, facts & EXACTLY_FUNCTION ? 1 : 0, env);
    if (facts & HAVE_TYPE)
        f(jl_type_typename, facts & EXACTLY_TYPE ? 1 : 0, env);
    if (facts & (HAVE_KWCALL | EXACTLY_KWCALL))
        f(jl_kwcall_type->name, facts & EXACTLY_KWCALL ? 1 : 0, env);
    f(jl_any_type->name, facts & EXACTLY_ANY ? 1 : 0, env);
    return 1;
}


int jl_foreach_reachable_mtable(int (*visit)(jl_methtable_t *mt, void *env), jl_array_t *mod_array, void *env)
{
    if (!visit(jl_method_table, env))
        return 0;

    if (!mod_array)
        return 1;

    arraylist_t workqueue;
    arraylist_new(&workqueue, 0);

    // Add initial toplevel modules to workqueue
    for (size_t i = 0; i < jl_array_nrows(mod_array); i++) {
        jl_module_t *m = (jl_module_t*)jl_array_ptr_ref(mod_array, i);
        assert(jl_is_module(m));
        if (m->parent == m) // some toplevel modules (really just Base) aren't actually
            arraylist_push(&workqueue, m);
    }

    int result = 1;

    while (workqueue.len > 0) {
        jl_module_t *current_m = (jl_module_t*)arraylist_pop(&workqueue);
        JL_GC_PROMISE_ROOTED(current_m);

        jl_svec_t *table = jl_atomic_load_relaxed(&current_m->bindings);
        for (size_t i = 0; i < jl_svec_len(table); i++) {
            jl_binding_t *b = (jl_binding_t*)jl_svecref(table, i);
            if ((void*)b == jl_nothing)
                break;
            jl_sym_t *name = b->globalref->name;
            jl_value_t *v = jl_get_latest_binding_value_if_const(b);
            if (v) {
                if (jl_is_module(v)) {
                    jl_module_t *child = (jl_module_t*)v;
                    if (child != current_m && child->parent == current_m && child->name == name) {
                        // this is the original/primary binding for the submodule
                        arraylist_push(&workqueue, child);
                    }
                }
                else if (jl_is_mtable(v)) {
                    jl_methtable_t *mt = (jl_methtable_t*)v;
                    if (mt && mt != jl_method_table && mt->module == current_m && mt->name == name) {
                        if (!visit(mt, env)) {
                            result = 0;
                            goto cleanup;
                        }
                    }
                }
            }
            table = jl_atomic_load_relaxed(&current_m->bindings);
        }
    }

cleanup:
    arraylist_free(&workqueue);
    return result;
}

jl_value_t *jl_typeinf_func JL_GLOBALLY_ROOTED = NULL;
jl_value_t *jl_compile_and_emit_func JL_GLOBALLY_ROOTED = NULL;
JL_DLLEXPORT size_t jl_typeinf_world = 1;

// Force Compiler (and staticdata serialization) not to throw away Julia IR,
// even when it is not needed for inlining, etc. - intended for debugging only
static _Atomic(int8_t) jl_type_infer_preserve_ir = 0;

JL_DLLEXPORT int8_t jl_get_type_infer_preserve_ir(void)
{
    return jl_atomic_load_relaxed(&jl_type_infer_preserve_ir);
}

JL_DLLEXPORT void jl_set_type_infer_preserve_ir(int8_t v)
{
    jl_atomic_store_relaxed(&jl_type_infer_preserve_ir, v);
}

// Set by precompile workers around `Base.include` so non-inlineable inferred
// IR is retained on `CodeInstance.inferred` through the irgen phase instead of
// being discarded. `jl_finalize_precompile_inferred` nulls them before save
// (with a backstop in jl_queue_for_serialization).
static _Atomic(int8_t) jl_precompile_keep_ir = 0;

JL_DLLEXPORT int8_t jl_get_precompile_keep_ir(void)
{
    return jl_atomic_load_relaxed(&jl_precompile_keep_ir);
}

JL_DLLEXPORT void jl_set_precompile_keep_ir(int8_t v)
{
    jl_atomic_store_relaxed(&jl_precompile_keep_ir, v);
}

JL_DLLEXPORT void jl_set_typeinf_func(jl_value_t *f)
{
    jl_typeinf_func = (jl_value_t*)f;
    jl_typeinf_world = jl_get_tls_world_age();
}

JL_DLLEXPORT void jl_set_compile_and_emit_func(jl_value_t *f)
{
    jl_compile_and_emit_func = (jl_value_t*)f;
}

static int very_general_type(jl_value_t *t)
{
    return (t == (jl_value_t*)jl_any_type || jl_types_equal(t, (jl_value_t*)jl_type_type));
}

jl_value_t *jl_nth_slot_type(jl_value_t *sig, size_t i) JL_NOTSAFEPOINT
{
    sig = jl_unwrap_unionall(sig);
    size_t len = jl_nparams(sig);
    if (i < len-1)
        return jl_tparam(sig, i);
    jl_value_t *p = jl_tparam(sig, len-1);
    if (jl_is_vararg(p))
        p = jl_unwrap_vararg(p);
    return p;
}

// if concrete_match returns false, the sig may specify `Type{T::DataType}`, while the `tt` contained DataType
// in this case, subtyping is wrong, and this may not actually match at runtime
// since it may instead match any kind of `Type{T::Type}`
//static int concrete_match(jl_tupletype_t *tt, jl_value_t *sig)
//{
//    size_t i, np;
//    for (i = 0, np = jl_nparams(tt); i < np; i++) {
//        jl_value_t *elt = jl_tparam(tt, i);
//        jl_value_t *decl_i = jl_nth_slot_type((jl_value_t*)sig, i);
//        if (jl_is_kind(elt)) {
//            // check whether this match may be exact at runtime
//            if (!jl_subtype(elt, decl_i))
//                return 0;
//        }
//    }
//    return 1;
//}

static jl_value_t *inst_varargp_in_env(jl_value_t *decl, jl_svec_t *sparams)
{
    jl_value_t *unw = jl_unwrap_unionall(decl);
    jl_value_t *vm = jl_tparam(unw, jl_nparams(unw) - 1);
    assert(jl_is_vararg(vm));
    int nsp = jl_svec_len(sparams);
    if (nsp > 0 && jl_has_free_typevars(vm)) {
        JL_GC_PUSH1(&vm);
        assert(jl_subtype_env_size(decl) == nsp);
        vm = jl_instantiate_type_in_env(vm, (jl_unionall_t*)decl, jl_svec_data(sparams));
        assert(jl_is_vararg(vm));
        // rewrap_unionall(lastdeclt, sparams) if any sparams isa TypeVar
        // for example, `Tuple{Vararg{Union{Nothing,Int,Val{T}}}} where T`
        // and the user called it with `Tuple{Vararg{Union{Nothing,Int},N}}`, then T is unbound
        jl_value_t **sp = jl_svec_data(sparams);
        while (jl_is_unionall(decl)) {
            jl_tvar_t *v = (jl_tvar_t*)*sp;
            if (jl_is_typevar(v)) {
                // must unwrap and re-wrap Vararg object explicitly here since jl_type_unionall handles it differently
                jl_value_t *T = ((jl_vararg_t*)vm)->T;
                jl_value_t *N = ((jl_vararg_t*)vm)->N;
                int T_has_tv = T && jl_has_typevar(T, v);
                int N_has_tv = N && jl_has_typevar(N, v); // n.b. JL_VARARG_UNBOUND check means this should be false
                assert(!N_has_tv || N == (jl_value_t*)v);
                vm = T_has_tv ? jl_type_unionall(v, T) : T;
                if (N_has_tv)
                    N = NULL;
                vm = (jl_value_t*)jl_wrap_vararg(vm, N, 1, 0); // this cannot throw for these inputs
            }
            sp++;
            decl = ((jl_unionall_t*)decl)->body;
            nsp--;
        }
        assert(nsp == 0);
        JL_GC_POP();
    }
    return vm;
}

static jl_value_t *ml_matches_i(jl_methtable_t *mt, jl_methcache_t *mc,
                              jl_tupletype_t *type, int lim, int include_ambiguous,
                              int intersections, size_t world, int cache_result,
                              size_t *min_valid, size_t *max_valid, int *ambig);

// The query world (compiler §4.11) for method matching: on the Julia-driven path
// Base.code_typed -> Base.invoke_in_world(w, ...) -> jl_call_in_typeinf_world ->
// Compiler.typeinf_code the task world is frozen at jl_typeinf_world, strictly below every
// runtime entry, so the match's membership questions would see no members. ml_matches always
// receives the query's world as an argument, so the whole match runs in it. A throw restores
// the task's query world in the handler that catches it, as it restores the world age.
static jl_value_t *ml_matches(jl_methtable_t *mt, jl_methcache_t *mc,
                              jl_tupletype_t *type, int lim, int include_ambiguous,
                              int intersections, size_t world, int cache_result,
                              size_t *min_valid, size_t *max_valid, int *ambig)
{
    size_t last_mq = jl_set_membership_query_world(world);
    jl_value_t *r = ml_matches_i(mt, mc, type, lim, include_ambiguous, intersections,
                                 world, cache_result, min_valid, max_valid, ambig);
    jl_set_membership_query_world(last_mq);
    return r;
}

// get the compilation signature specialization for this method
// ---- the attribute of a trait argument (compiler §6.3) ----
// Which of a method's arguments are compiled once at their declared type (`Dynamic`) rather than for
// the value's type (`Compiled`). A trait argument is one typed by a set or an intersection of sets,
// its type naming the method's variables or not (`X{T}` is compiled once at `X{T₀}`), not by a `where`
// variable (whose binding is the value's type).

// the package a module belongs to: its root module; a module under `Main`, as a script makes one,
// is its own
static jl_module_t *module_package(jl_module_t *m) JL_NOTSAFEPOINT
{
    jl_module_t *p = m;
    while (p->parent != p)
        p = p->parent;
    return p == jl_main_module ? m : p;
}

static int is_trait_arg_type(jl_value_t *decl_i) JL_NOTSAFEPOINT
{
    // a set or an intersection, also where its type names the method's variables (`X{T}`): compiled
    // once at its instance with those variables' values (compiler §6.2; C-54). A `where` variable
    // bounded by a set is the value's own type.
    if (jl_is_typevar(decl_i))
        return 0;
    jl_value_t *u = jl_unwrap_unionall(decl_i);
    return jl_is_datatype(u) && (((jl_datatype_t*)u)->name->open_set || jl_is_all_type(u));
}

// the default attribute of the sets that type a trait argument (step 4): 0 none, 1 `Compiled`,
// 2 `Dynamic`; for an intersection, the components' where they agree, else none
static int set_default_attribute(jl_value_t *decl_i, jl_module_t **setmod) JL_NOTSAFEPOINT
{
    jl_datatype_t *u = (jl_datatype_t*)jl_unwrap_unionall(decl_i);
    if (!jl_is_all_type((jl_value_t*)u)) {
        *setmod = u->name->module;
        return u->name->default_attribute;
    }
    jl_value_t *tup = jl_tparam0(u);
    int d = -1;
    for (size_t k = 0; jl_is_tuple_type(tup) && k < jl_nparams(tup); k++) {
        jl_value_t *c = jl_unwrap_unionall(jl_tparam(tup, k));
        if (!jl_is_datatype(c))
            return 0;
        int dk = ((jl_datatype_t*)c)->name->default_attribute;
        if (d == -1) {
            d = dk;
            *setmod = ((jl_datatype_t*)c)->name->module;
        }
        else if (d != dk)
            return 0;
    }
    return d < 0 ? 0 : d;
}

// whether a type is Julia's own or the set's package's, with every type among its parameters,
// recursively (a value parameter, a number or a symbol, is the type's own): the set's default does
// not apply to a value of such a type, since such types are few and fixed. A parametric type that
// carries another package's type (`FunctionDynamics{typeof(f)}`, `Base.Fix1{typeof(f)}`) is not.
static int type_is_own_(jl_value_t *t, jl_module_t *setpkg, int depth) JL_NOTSAFEPOINT
{
    if (depth > 8)
        return 0;
    if (jl_is_typevar(t))
        return type_is_own_(((jl_tvar_t*)t)->ub, setpkg, depth + 1);
    if (jl_is_unionall(t))
        return type_is_own_(jl_unwrap_unionall(t), setpkg, depth);
    if (jl_is_uniontype(t))
        return type_is_own_(((jl_uniontype_t*)t)->a, setpkg, depth + 1) &&
               type_is_own_(((jl_uniontype_t*)t)->b, setpkg, depth + 1);
    if (jl_is_vararg(t)) {
        jl_value_t *vt = jl_unwrap_vararg(t);
        return vt == NULL || type_is_own_(vt, setpkg, depth + 1);
    }
    if (!jl_is_datatype(t))
        return 1; // a value parameter
    jl_datatype_t *dt = (jl_datatype_t*)t;
    jl_module_t *p = module_package(dt->name->module);
    if (!(p == jl_core_module || p == jl_base_module || (setpkg != NULL && p == setpkg)))
        return 0;
    for (size_t k = 0; k < jl_nparams(dt); k++) {
        if (!type_is_own_(jl_tparam(dt, k), setpkg, depth + 1))
            return 0;
    }
    return 1;
}
static int type_is_own(jl_value_t *elt, jl_module_t *setmod) JL_NOTSAFEPOINT
{
    return type_is_own_(elt, setmod != NULL ? module_package(setmod) : NULL, 0);
}

// The arguments of `m` that a call with argument types `tt` compiles once (compiler §6.3, steps 3 to
// 5, resolved at the method): the method's `@nospecialize` and `@specialize` at the argument; the
// module's `dynamic` option; the set's default, except for a value of Julia's own types or the set's
// package's; the module's `first_seen` option, `Dynamic` at a dynamic call. Returned as the
// method's `nospecialize` bits with these decisions applied. Casts and the attribute a value carries
// (steps 1, 2) override it: `force_mask` names the arguments they decide, `force_dynamic` which are
// `Dynamic`.
static uint32_t effective_nospecialize(jl_method_t *m, jl_tupletype_t *tt, int static_call,
                                       uint32_t force_mask, uint32_t force_dynamic) JL_NOTSAFEPOINT
{
    uint32_t nospec = m->nospecialize;
    if (m->generator || !jl_has_open_set_in(m->sig))
        return nospec;
    jl_value_t *decl = m->sig;
    size_t nargs = m->nargs, np = jl_nparams(tt);
    int opts = m->module->trait_options;
    for (size_t i = 1; i < nargs && i < np && i <= 32; i++) {
        if (m->isva && i == nargs - 1)
            break; // a varargs tail is compiled as Julia compiles it
        jl_value_t *decl_i = jl_nth_slot_type(decl, i);
        if (!is_trait_arg_type(decl_i))
            continue;
        uint32_t bit = 1u << (i - 1);
        int dyn;
        if (force_mask & bit)
            dyn = (force_dynamic & bit) != 0;                     // 1, 2: a cast, or carried
        else if (nospec & bit)
            continue;                                             // 3: `@nospecialize`
        else if (m->specialize & bit)
            dyn = 0;                                              // 3: `@specialize`
        else if (opts & 4)
            dyn = 1;                                              // 3: the module's `dynamic`
        else {
            jl_module_t *setmod = NULL;
            int d = set_default_attribute(decl_i, &setmod);
            jl_value_t *elt = jl_tparam(tt, i);
            if (d != 0 && !jl_is_vararg(elt) && !type_is_own(elt, setmod))
                dyn = d == 2;                                     // 4: the set's default
            else // 5: `first_seen`: `Compiled` where the caller knows the value's type; else `Compiled`
                dyn = (opts & 8) && !(static_call && !jl_is_vararg(elt) && jl_is_concrete_type(elt));
        }
        if (dyn)
            nospec |= bit;
        else
            nospec &= ~bit;
    }
    return nospec;
}

JL_DLLEXPORT int jl_is_trait_arg_type(jl_value_t *t) JL_NOTSAFEPOINT
{
    return is_trait_arg_type(t);
}

// the trait arguments of `m` (as nospecialize bits): a compileable signature may have the declared
// type there whichever attribute decided it
static uint32_t trait_arg_bits(jl_method_t *m) JL_NOTSAFEPOINT
{
    if (m->generator || !jl_has_open_set_in(m->sig))
        return 0;
    uint32_t bits = 0;
    for (size_t i = 1; i < m->nargs && i <= 32; i++) {
        if (m->isva && i == m->nargs - 1)
            break;
        if (is_trait_arg_type(jl_nth_slot_type(m->sig, i)))
            bits |= 1u << (i - 1);
    }
    return bits;
}

static void jl_compilation_sig(
    jl_tupletype_t *const tt, // the original tupletype of the call (or DataType from precompile)
    jl_svec_t *sparams,
    jl_method_t *definition,
    intptr_t max_varargs,
    uint32_t nospec, // the method's nospecialize bits with the trait arguments' attributes applied
    // output:
    jl_svec_t **const newparams JL_REQUIRE_ROOTED_SLOT)
{
    assert(jl_is_tuple_type(tt));
    jl_value_t *decl = definition->sig;
    size_t nargs = definition->nargs; // == jl_nparams(jl_unwrap_unionall(decl));
    size_t nspec = max_varargs + nargs;

    if (definition->generator) {
        // staged functions aren't optimized
        // so assume the caller was intelligent about calling us
        return;
    }

    if (decl == (jl_value_t*)jl_anytuple_type && jl_atomic_load_relaxed(&definition->unspecialized)) {
        *newparams = jl_anytuple_type->parameters; // handle builtin methods
        return;
    }

    // some early sanity checks
    size_t i, np = jl_nparams(tt);
    switch (jl_va_tuple_kind((jl_datatype_t*)decl)) {
    case JL_VARARG_NONE:
        if (jl_is_va_tuple(tt))
            // odd
            return;
        if (np != nargs)
            // there are not enough input parameters to make this into a compilation sig
            return;
        break;
    case JL_VARARG_INT:
    case JL_VARARG_BOUND:
        if (jl_is_va_tuple(tt))
            // the length needed is not known, but required for compilation
            return;
        if (np < nargs - 1)
            // there are not enough input parameters to make this into a compilation sig
            return;
        break;
    case JL_VARARG_UNBOUND:
        if (np < nspec && jl_is_va_tuple(tt))
            // there are insufficient given parameters for jl_isa_compileable_sig now to like this type
            // (there were probably fewer methods defined when we first selected this signature, or
            //  the max varargs limit was not reached indicating the type is already fully-specialized)
            return;
        break;
    }

    jl_value_t *type_i = NULL;
    JL_GC_PUSH1(&type_i);
    for (i = 0; i < np; i++) {
        jl_value_t *elt = jl_tparam(tt, i);
        if (jl_is_vararg(elt))
            elt = jl_unwrap_vararg(elt);
        jl_value_t *decl_i = jl_nth_slot_type(decl, i);
        type_i = jl_rewrap_unionall(decl_i, decl);
        size_t i_arg = (i < nargs - 1 ? i : nargs - 1);

        if (jl_is_kind(type_i)) {
            // if we can prove the match was against the kind (not a Type)
            // we want to put that in the cache instead
            if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
            elt = type_i;
            jl_svecset(*newparams, i, elt);
        }
        else if (jl_is_type_type(elt)) {
            // if the declared type was not Any or Union{Type, ...},
            // then the match must been with the kind (e.g. UnionAll or DataType)
            // and the result of matching the type signature
            // needs to be restricted to the concrete type 'kind'
            jl_value_t *kind = jl_typeof(jl_tparam0(elt));
            if (jl_subtype(kind, type_i) && !jl_subtype((jl_value_t*)jl_type_type, type_i)) {
                // if we can prove the match was against the kind (not a Type)
                // it's simpler (and thus better) to put that cache instead
                if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
                elt = kind;
                jl_svecset(*newparams, i, elt);
            }
        }
        else if (jl_is_kind(elt)) {
            // not triggered for isdispatchtuple(tt), this attempts to handle
            // some cases of adapting a random signature into a compilation signature
            // if we get a kind, where we don't expect to accept one, widen it to something more expected (Type{T})
            if (!(jl_subtype(elt, type_i) && !jl_subtype((jl_value_t*)jl_type_type, type_i))) {
                if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
                elt = (jl_value_t*)jl_type_type;
                jl_svecset(*newparams, i, elt);
            }
        }


        if (jl_is_kind(elt)) {
            // kind slots always need guard entries (checking for subtypes of Type)
            continue;
        }

        if (i_arg > 0 && i_arg <= sizeof(definition->nospecialize) * 8 &&
                (nospec & (1 << (i_arg - 1)))) {
            if (!jl_has_free_typevars(decl_i) && !jl_is_kind(decl_i)) {
                if (decl_i != elt) {
                    if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
                    // n.b. it is possible here that !(elt <: decl_i), if elt was something unusual from intersection
                    // so this might narrow the result slightly, though still being compatible with the declared signature
                    jl_svecset(*newparams, i, (jl_value_t*)decl_i);
                }
                continue;
            }
            // a trait argument whose type names the method's variables (`X{T}`): compiled once at that
            // instance with the variables' values from the match, the erased key (compiler §6.2; C-54)
            if (is_trait_arg_type(decl_i) && sparams != NULL && jl_svec_len(sparams) > 0 && jl_is_unionall(decl)) {
                type_i = jl_instantiate_type_in_env(decl_i, (jl_unionall_t*)decl, jl_svec_data(sparams));
                if (!jl_has_free_typevars(type_i)) {
                    if (type_i != elt) {
                        if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
                        jl_svecset(*newparams, i, type_i);
                    }
                    continue;
                }
                type_i = jl_rewrap_unionall(decl_i, decl);
            }
        }

        if (jl_types_equal(elt, (jl_value_t*)jl_type_type)) { // elt == Type{T} where T
            // not triggered for isdispatchtuple(tt), this attempts to handle
            // some cases of adapting a random signature into a compilation signature
        }
        else if (!jl_is_datatype(elt) && jl_subtype(elt, (jl_value_t*)jl_type_type)) { // elt <: Type{T}
            // not triggered for isdispatchtuple(tt), this attempts to handle
            // some cases of adapting a random signature into a compilation signature
            if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
            jl_svecset(*newparams, i, jl_type_type);
        }
        else if (jl_is_type_type(elt)) { // elt isa Type{T}
            if (!jl_has_free_typevars(decl_i) && very_general_type(type_i)) {
                /*
                  Here's a fairly simple heuristic: if this argument slot's
                  declared type is general (Type or Any),
                  then don't specialize for every Type that got passed.

                  Since every type x has its own type Type{x}, this would be
                  excessive specialization for an Any slot.

                  This may require guard entries due to other potential matches.
                  In particular, TypeConstructors are problematic because they can
                  be alternate representations of any type. Extensionally, TC == TC.body,
                  but typeof(TC) != typeof(TC.body). This creates an ambiguity:
                  Type{TC} is type-equal to Type{TC.body}, yet a slot
                  x::TypeConstructor matches the first but not the second, while
                  also matching all other TypeConstructors. This means neither
                  Type{TC} nor TypeConstructor is more specific.

                  But don't apply this heuristic if the argument is called (issue #36783).
                */
                int iscalled = i_arg > 0 && i_arg <= 8 && (definition->called & (1 << (i_arg - 1)));
                if (!iscalled) {
                    if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
                    jl_svecset(*newparams, i, jl_type_type);
                }
            }
            else if (jl_is_type_type(jl_tparam0(elt)) &&
                     // try to give up on specializing type parameters for Type{Type{Type{...}}}
                     (jl_is_type_type(jl_tparam0(jl_tparam0(elt))) || !jl_has_free_typevars(decl_i))) {
                /*
                  actual argument was Type{...}, we computed its type as
                  Type{Type{...}}. we like to avoid unbounded nesting here, so
                  compile (and hopefully cache) the signature as Type{T},
                  unless something more specific like Type{Type{Int32}} was
                  actually declared. this can be determined using a type
                  intersection.
                */
                if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
                if (i < nargs || !definition->isva) {
                    jl_value_t *di = jl_type_intersection(type_i, (jl_value_t*)jl_type_type);
                    assert(di != (jl_value_t*)jl_bottom_type);
                    // issue #11355: DataType has a UID and so would take precedence in the cache
                    if (jl_is_kind(di))
                        jl_svecset(*newparams, i, (jl_value_t*)jl_type_type);
                    else
                        jl_svecset(*newparams, i, di);
                }
                else {
                    jl_svecset(*newparams, i, (jl_value_t*)jl_type_type);
                }
            }
        }

        int notcalled_func = (i_arg > 0 && i_arg <= 8 && !(definition->called & (1 << (i_arg - 1))) &&
                              !jl_has_free_typevars(decl_i) &&
                              jl_subtype(elt, (jl_value_t*)jl_function_type));
        if (notcalled_func && (jl_subtype((jl_value_t*)jl_function_type, type_i))) {
            // and attempt to despecialize types marked as a supertype of Function (i.e.
            // Function, Callable, Any, or a Union{Function, T})
            // when called with a subtype of Function but is not called
            if (!*newparams) *newparams = jl_svec_copy(tt->parameters);
            jl_svecset(*newparams, i, (jl_value_t*)jl_function_type);
        }
    }

    // for varargs methods, only specialize up to max_args.
    // in general, here we want to find the biggest type that's not a
    // supertype of any other method signatures. so far we are conservative
    // and the types we find should be bigger.
    if (np >= nspec && jl_va_tuple_kind((jl_datatype_t*)decl) == JL_VARARG_UNBOUND) {
        if (!*newparams) *newparams = tt->parameters;
        if (max_varargs > 0 && nspec >= 2) {
            type_i = jl_svecref(*newparams, nspec - 2);
        }
        else {
            // If max varargs is zero, always specialize to (Any...) since
            // there is no preceding parameter to use for `type_i`
            type_i = jl_bottom_type;
        }
        // if all subsequent arguments are subtypes of type_i, specialize
        // on that instead of decl. for example, if decl is
        // (Any...)
        // and type is
        // (Symbol, Symbol, Symbol)
        // then specialize as (Symbol...), but if type is
        // (Symbol, Int32, Expr)
        // then specialize as (Any...)
        size_t j = nspec - 1;
        int all_are_subtypes = 1;
        for (; j < jl_svec_len(*newparams); j++) {
            jl_value_t *paramj = jl_svecref(*newparams, j);
            if (jl_is_vararg(paramj))
                paramj = jl_unwrap_vararg(paramj);
            if (!jl_subtype(paramj, type_i)) {
                all_are_subtypes = 0;
                break;
            }
        }
        if (all_are_subtypes) {
            // avoid Vararg{Type{Type{...}}}
            if (jl_is_type_type(type_i) && jl_is_type_type(jl_tparam0(type_i)))
                type_i = (jl_value_t*)jl_type_type;
            type_i = (jl_value_t*)jl_wrap_vararg(type_i, (jl_value_t*)NULL, 1, 0); // this cannot throw for these inputs
        }
        else {
            type_i = inst_varargp_in_env(decl, sparams);
        }
        jl_svec_t *limited = jl_alloc_svec(nspec);
        size_t i;
        for (i = 0; i < nspec - 1; i++) {
            jl_svecset(limited, i, jl_svecref(*newparams, i));
        }
        jl_svecset(limited, i, type_i);
        *newparams = limited;
    }
    JL_GC_POP();
}

// compute whether this type signature is a possible return value from jl_compilation_sig given a concrete-type for `tt`
JL_DLLEXPORT int jl_isa_compileable_sig(
    jl_tupletype_t *type,
    jl_svec_t *sparams,
    jl_method_t *definition)
{
    jl_value_t *decl = definition->sig;

    if (!jl_is_datatype(type) || jl_has_free_typevars((jl_value_t*)type))
        return 0;
    if (definition->sig == (jl_value_t*)jl_anytuple_type && jl_atomic_load_relaxed(&definition->unspecialized))
        return jl_egal((jl_value_t*)type, definition->sig); // handle builtin methods

    size_t i, np = jl_nparams(type);
    size_t nargs = definition->nargs; // == jl_nparams(jl_unwrap_unionall(decl));
    if (np == 0)
        return nargs == 0;
    uint32_t trait_bits = trait_arg_bits(definition);

    if (definition->generator) {
        // staged functions aren't optimized
        // so assume the caller was intelligent about calling us
        return (definition->isva ? np >= nargs - 1 : np == nargs) && type->isdispatchtuple;
    }

    // for varargs methods, only specialize up to max_args (>= nargs + 1).
    // in general, here we want to find the biggest type that's not a
    // supertype of any other method signatures. so far we are conservative
    // and the types we find should be bigger.
    if (definition->isva) {
        unsigned nspec_min = nargs + 1; // min number of arg values (including tail vararg)
        unsigned nspec_max = INT32_MAX; // max number of arg values (including tail vararg)
        // try to refine estimate of min and max
        uint8_t heuristic_used = 0;
        nspec_max = nspec_min = nargs + get_max_varargs(definition, &heuristic_used);
        if (heuristic_used)
            nspec_max = INT32_MAX; // new methods may be added, increasing nspec_min later
        int isunbound = (jl_va_tuple_kind((jl_datatype_t*)decl) == JL_VARARG_UNBOUND);
        if (jl_is_vararg(jl_tparam(type, np - 1))) {
            if (!isunbound || np < nspec_min || np > nspec_max)
                return 0;
        }
        else {
            if (np < nargs - 1 || (isunbound && np >= nspec_max))
                return 0;
        }
    }
    else if (np != nargs || jl_is_vararg(jl_tparam(type, np - 1))) {
        return 0;
    }

    jl_value_t *type_i = NULL;
    JL_GC_PUSH1(&type_i);
    for (i = 0; i < np; i++) {
        jl_value_t *elt = jl_tparam(type, i);
        size_t i_arg = (i < nargs - 1 ? i : nargs - 1);

        if (jl_is_vararg(elt)) {
            type_i = inst_varargp_in_env(decl, sparams);
            if (jl_has_free_typevars(type_i)) {
                JL_GC_POP();
                return 0; // something went badly wrong?
            }
            if (jl_egal(elt, type_i))
                continue; // elt could be chosen by inst_varargp_in_env for these sparams
            elt = jl_unwrap_vararg(elt);
            if (jl_is_type_type(elt) && jl_is_type_type(jl_tparam0(elt))) {
                JL_GC_POP();
                return 0; // elt would be set equal to jl_type_type instead
            }
            // else, elt also needs to meet the usual rules
        }

        jl_value_t *decl_i = jl_nth_slot_type(decl, i);
        type_i = jl_rewrap_unionall(decl_i, decl);

        if (i_arg > 0 && i_arg <= sizeof(definition->nospecialize) * 8 &&
                (definition->nospecialize & (1 << (i_arg - 1))) &&
                !(trait_bits & (1u << (i_arg - 1)))) { // at a trait argument the attribute decides
            if (!jl_has_free_typevars(decl_i) && !jl_is_kind(decl_i)) {
                if (jl_egal(elt, decl_i))
                    continue;
                JL_GC_POP();
                return 0;
            }
        }
        // a trait argument compiled once has its declared type, whichever attribute decided it
        if (i_arg > 0 && i_arg <= 32 && (trait_bits & (1u << (i_arg - 1))) && jl_egal(elt, decl_i))
            continue;
        // or, where that type names the method's variables, its instance with their values (C-54)
        if (i_arg > 0 && i_arg <= 32 && (trait_bits & (1u << (i_arg - 1))) && jl_has_free_typevars(decl_i) &&
                is_trait_arg_type(elt) && !jl_has_free_typevars(elt) &&
                ((jl_datatype_t*)jl_unwrap_unionall(elt))->name == ((jl_datatype_t*)jl_unwrap_unionall(decl_i))->name &&
                jl_subtype(elt, type_i))
            continue;

        if (jl_is_kind(elt)) {
            // kind slots always get guard entries (checking for subtypes of Type)
            if (jl_subtype(elt, type_i) && !jl_subtype((jl_value_t*)jl_type_type, type_i))
                continue;
            // TODO: other code paths that could reach here?
            JL_GC_POP();
            return 0;
        }
        else if (jl_is_kind(type_i)) {
            JL_GC_POP();
            return 0;
        }

        if (jl_is_type_type(jl_unwrap_unionall(elt))) {
            int iscalled = (i_arg > 0 && i_arg <= 8 && (definition->called & (1 << (i_arg - 1)))) ||
                           jl_has_free_typevars(decl_i);
            if (jl_types_equal(elt, (jl_value_t*)jl_type_type)) {
                if (!iscalled && very_general_type(type_i))
                    continue;
                if (i >= nargs && definition->isva)
                    continue;
                JL_GC_POP();
                return 0;
            }
            if (!iscalled && very_general_type(type_i)) {
                JL_GC_POP();
                return 0;
            }
            if (!jl_is_datatype(elt)) {
                JL_GC_POP();
                return 0;
            }

            // if the declared type was not Any or Union{Type, ...},
            // then the match must been with kind, such as UnionAll or DataType,
            // and the result of matching the type signature
            // needs to be corrected to the concrete type 'kind' (and not to Type)
            jl_value_t *kind = jl_typeof(jl_tparam0(elt));
            if (kind == jl_bottom_type) {
                JL_GC_POP();
                return 0; // Type{Union{}} gets normalized to typeof(Union{})
            }
            if (jl_subtype(kind, type_i) && !jl_subtype((jl_value_t*)jl_type_type, type_i)) {
                JL_GC_POP();
                return 0; // gets turned into a kind
            }

            else if (jl_is_type_type(jl_tparam0(elt)) &&
                     // give up on specializing static parameters for Type{Type{Type{...}}}
                     (jl_is_type_type(jl_tparam0(jl_tparam0(elt))) || !jl_has_free_typevars(decl_i))) {
                /*
                  actual argument was Type{...}, we computed its type as
                  Type{Type{...}}. we must avoid unbounded nesting here, so
                  cache the signature as Type{T}, unless something more
                  specific like Type{Type{Int32}} was actually declared.
                  this can be determined using a type intersection.
                */
                if (i < nargs || !definition->isva) {
                    type_i = jl_type_intersection(type_i, (jl_value_t*)jl_type_type);
                    assert(type_i != (jl_value_t*)jl_bottom_type);
                    if (jl_is_kind(type_i)) {
                        JL_GC_POP();
                        return 0;
                    }
                    else if (!jl_types_equal(type_i, elt)) {
                        JL_GC_POP();
                        return 0;
                    }
                    continue;
                }
                else {
                    JL_GC_POP();
                    return 0;
                }
            }
            continue;
        }

        int notcalled_func = (i_arg > 0 && i_arg <= 8 && !(definition->called & (1 << (i_arg - 1))) &&
                              !jl_has_free_typevars(decl_i) &&
                              jl_subtype(elt, (jl_value_t*)jl_function_type));
        if (notcalled_func && jl_subtype((jl_value_t*)jl_function_type, type_i)) {
            // and attempt to despecialize types marked as a supertype of Function (i.e.
            // Function, Callable, Any, or a Union{Function, T})
            if (elt == (jl_value_t*)jl_function_type)
                continue;
            JL_GC_POP();
            return 0;
        }

        if (!jl_is_concrete_type(elt)) {
            JL_GC_POP();
            return 0;
        }
    }
    JL_GC_POP();
    return 1;
}


static int concretesig_equal(jl_value_t *tt, jl_value_t *simplesig) JL_NOTSAFEPOINT
{
    jl_value_t **types = jl_svec_data(((jl_datatype_t*)tt)->parameters);
    jl_value_t **sigs = jl_svec_data(((jl_datatype_t*)simplesig)->parameters);
    size_t i, lensig = jl_nparams(simplesig);
    assert(lensig == jl_nparams(tt));
    assert(lensig > 0 && !jl_is_vararg(jl_tparam(simplesig, lensig - 1)));
    for (i = 0; i < lensig; i++) {
        jl_value_t *decl = sigs[i];
        jl_value_t *a = types[i];
        if (a != decl && decl != (jl_value_t*)jl_any_type) {
            if (!(jl_is_type_type(a) && jl_typeof(jl_tparam0(a)) == decl))
                return 0;
        }
    }
    return 1;
}

// if available, returns a TypeMapEntry in the "leafcache" that matches `tt` (by type-equality) and is valid during `world`
static inline jl_typemap_entry_t *lookup_leafcache(jl_genericmemory_t *leafcache JL_PROPAGATES_ROOT, jl_value_t *tt, size_t world) JL_NOTSAFEPOINT
{
    jl_typemap_entry_t *entry = (jl_typemap_entry_t*)jl_eqtable_get(leafcache, (jl_value_t*)tt, NULL);
    if (entry) {
        // search tail of the linked-list (including the returned entry) for an entry intersecting world
        //
        // n.b. this entire chain is type-equal to tt (by construction), so it is unnecessary to call `tt<:entry->sig`
        do {
            if (jl_atomic_load_relaxed(&entry->min_world) <= world && world <= jl_atomic_load_relaxed(&entry->max_world)) {
                if (entry->simplesig == (void*)jl_nothing || concretesig_equal(tt, (jl_value_t*)entry->simplesig))
                    return entry;
            }
            entry = jl_atomic_load_relaxed(&entry->next);
        } while ((jl_value_t*)entry != jl_nothing);
    }
    return NULL;
}
jl_method_instance_t *cache_method(
        jl_methtable_t *mt, jl_methcache_t *mc, _Atomic(jl_typemap_t*) *cache, jl_value_t *parent JL_PROPAGATES_ROOT,
        jl_tupletype_t *tt, // the original tupletype of the signature
        jl_method_t *definition,
        size_t world, size_t min_valid, size_t max_valid,
        jl_svec_t *sparams,
        // set by callers that have already proven `tt` is absent from the cache
        // immediately before acquiring the lock and that no insertion can have
        // happened since, allowing the redundant re-check below to be skipped
        int tt_known_absent)
{
    // caller must hold the parent->writelock, which this releases
    // short-circuit (now that we hold the lock) if this entry is already present
    int8_t offs = mc ? jl_cachearg_offset() : 1;
    if (!tt_known_absent) { // scope block
        if (mc) {
            jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
            jl_typemap_entry_t *entry = lookup_leafcache(leafcache, (jl_value_t*)tt, world);
            if (entry) {
                if (mc) JL_UNLOCK(&mc->writelock);
                return entry->func.linfo;
            }
        }
        struct jl_typemap_assoc search = {(jl_value_t*)tt, world, NULL};
        jl_typemap_t *cacheentry = jl_atomic_load_relaxed(cache);
        assert(cacheentry != NULL);
        jl_typemap_entry_t *entry = jl_typemap_assoc_by_type(cacheentry, &search, offs, /*subtype*/1);
        if (entry && entry->func.value) {
            if (mc) JL_UNLOCK(&mc->writelock);
            return entry->func.linfo;
        }
    }

    jl_method_instance_t *newmeth = NULL;
    if (definition->sig == (jl_value_t*)jl_anytuple_type && definition != jl_opaque_closure_method && !definition->is_for_opaque_closure) {
        newmeth = jl_atomic_load_relaxed(&definition->unspecialized);
        if (newmeth != NULL) { // handle builtin methods de-specialization (for invoke, or if the global cache entry somehow gets lost)
            jl_tupletype_t *cachett = (jl_tupletype_t*)newmeth->specTypes;
            assert(cachett != jl_anytuple_type);
            jl_typemap_entry_t *newentry = jl_typemap_alloc(cachett, NULL, jl_emptysvec, (jl_value_t*)newmeth, min_valid, max_valid);
            JL_GC_PUSH1(&newentry);
            jl_typemap_insert(cache, parent, newentry, offs);
            if (mc)
                jl_method_cache_inserted();
            JL_GC_POP();
            if (mc) JL_UNLOCK(&mc->writelock);
            return newmeth;
        }
    }

    jl_value_t *temp = NULL;
    jl_value_t *temp2 = NULL;
    jl_value_t *temp3 = NULL;
    jl_svec_t *newparams = NULL;
    JL_GC_PUSH5(&temp, &temp2, &temp3, &newmeth, &newparams);

    // Consider if we can cache with the preferred compile signature
    // so that we can minimize the number of required cache entries.
    int cache_with_orig = 1;
    jl_tupletype_t *compilationsig = tt;
    intptr_t max_varargs = get_max_varargs(definition, NULL);
    // a dynamic call: the trait arguments' attributes resolved at the method (compiler §6.3). An
    // instance compiled once by an attribute is cached with the call's own types: the attribute may
    // differ for another member (a set's default skips Julia's own types and its package's).
    uint32_t nospec = effective_nospecialize(definition, tt, 0, 0, 0);
    int attr_widened = (nospec & ~definition->nospecialize & trait_arg_bits(definition)) != 0;
    jl_compilation_sig(tt, sparams, definition, max_varargs, nospec, &newparams);
    if (newparams) {
        temp2 = jl_apply_tuple_type(newparams, 1);
        // Now there may be a problem: the widened signature is more general
        // than just the given arguments, so it might conflict with another
        // definition that does not have cache instances yet. To fix this, we
        // may insert guard cache entries for all intersections of this
        // signature and definitions. Those guard entries will supersede this
        // one in conflicted cases, alerting us that there should actually be a
        // cache miss. Alternatively, we may use the original signature in the
        // cache, but use this return for compilation.
        //
        // In most cases `!jl_isa_compileable_sig(tt, sparams, definition)`,
        // although for some cases, (notably Varargs)
        // we might choose a replacement type that's preferable but not strictly better
        int issubty;
        temp = jl_type_intersection_env_s(temp2, (jl_value_t*)definition->sig, &newparams, &issubty);
        assert(temp != (jl_value_t*)jl_bottom_type); (void)temp;
        if (jl_egal((jl_value_t*)newparams, (jl_value_t*)sparams)) {
            cache_with_orig = !issubty;
            compilationsig = (jl_datatype_t*)temp2;
        }
        newparams = NULL;
    }
    // TODO: maybe assert(jl_isa_compileable_sig(compilationsig, sparams, definition));
    newmeth = jl_specializations_get_linfo(definition, (jl_value_t*)compilationsig, sparams);
    if (newmeth->cache_with_orig || attr_widened)
        cache_with_orig = 1;

    // Capture world counter at start to detect races
    size_t current_world = mc ? jl_atomic_load_acquire(&jl_world_counter) : ~(size_t)0;

    jl_tupletype_t *cachett = tt;
    jl_svec_t *guardsigs = jl_emptysvec;
    if (!cache_with_orig && mt) {
        // now examine what will happen if we chose to use this sig in the cache
        size_t min_valid2 = 1;
        size_t max_valid2 = ~(size_t)0;
        temp = ml_matches(mt, mc, compilationsig, MAX_UNSPECIALIZED_CONFLICTS, 1, 1, world, 0, &min_valid2, &max_valid2, NULL);
        int guards = 0;
        if (temp == jl_nothing) {
            cache_with_orig = 1;
        }
        else {
            int unmatched_tvars = 0;
            size_t i, l = jl_array_nrows(temp);
            for (i = 0; i < l; i++) {
                jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(temp, i);
                if (matc->method == definition)
                    continue;
                jl_svec_t *env = matc->sparams;
                int k, l;
                for (k = 0, l = jl_svec_len(env); k < l; k++) {
                    jl_value_t *env_k = jl_svecref(env, k);
                    if (jl_is_typevar(env_k) || jl_is_vararg(env_k)) {
                        unmatched_tvars = 1;
                        break;
                    }
                }
                if (unmatched_tvars || guards > MAX_UNSPECIALIZED_CONFLICTS) {
                    // if distinguishing a guard entry from the generalized signature
                    // would require matching type vars then bail out, since the
                    // method cache matching algorithm cannot do that.
                    //
                    // also bail if this requires too many guard entries
                    cache_with_orig = 1;
                    break;
                }
                guards++;
            }
        }
        if (!cache_with_orig && guards > 0) {
            // use guard entries as placeholders to prevent this cached method
            // from matching when another more specific definition also exists
            size_t i, l;
            guardsigs = jl_alloc_svec(guards);
            temp3 = (jl_value_t*)guardsigs;
            guards = 0;
            for (i = 0, l = jl_array_nrows(temp); i < l; i++) {
                jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(temp, i);
                jl_method_t *other = matc->method;
                if (other != definition) {
                    jl_svecset(guardsigs, guards, matc->spec_types);
                    guards++;
                    // alternative approach: insert sentinel entry
                    //jl_typemap_insert(cache, parent, (jl_tupletype_t*)matc->spec_types,
                    //        NULL, jl_emptysvec, /*guard*/NULL, jl_cachearg_offset(), other->min_world, other->max_world);
                }
            }
            assert(guards == jl_svec_len(guardsigs));
        }
        if (!cache_with_orig) {
            // determined above that there's no ambiguity in also using compilationsig as the cacheablesig
            min_valid = min_valid2;
            max_valid = max_valid2;
            cachett = compilationsig;
        }
        else {
            // do not revisit this decision
            newmeth->cache_with_orig = 1;
        }
    }

    int unconstrained_max = max_valid == ~(size_t)0;
    if (max_valid > current_world)
        max_valid = current_world;

    // now scan `cachett` and ensure that `Type{T}` in the cache will be matched exactly by `typeof(T)`
    // and also reduce the complexity of rejecting this entry in the cache
    // by replacing non-simple types with jl_any_type to build a new `type`
    // (for example, if the signature contains jl_function_type)
    // TODO: this is also related to how we should handle partial matches
    //       (which currently might miss detection of a MethodError)
    jl_tupletype_t *simplett = NULL;
    size_t i, np = jl_nparams(cachett);
    newparams = NULL;
    for (i = 0; i < np; i++) {
        jl_value_t *elt = jl_svecref(cachett->parameters, i);
        if (jl_is_vararg(elt)) {
        }
        else if (jl_is_type_type(elt)) {
            // TODO: if (!jl_is_singleton(elt)) ...
            jl_value_t *kind = jl_typeof(jl_tparam0(elt));
            if (!newparams) newparams = jl_svec_copy(cachett->parameters);
            jl_svecset(newparams, i, kind);
        }
        else if (!jl_is_concrete_type(elt)) { // for example, jl_function_type or jl_tuple_type
            if (!newparams) newparams = jl_svec_copy(cachett->parameters);
            jl_svecset(newparams, i, jl_any_type);
        }
    }
    if (newparams) {
        simplett = (jl_datatype_t*)jl_apply_tuple_type(newparams, 1);
        temp2 = (jl_value_t*)simplett;
    }

    // short-circuit if an existing entry is already present
    // that satisfies our requirements
    if (cachett != tt) {
        struct jl_typemap_assoc search = {(jl_value_t*)cachett, world, NULL};
        jl_typemap_entry_t *entry = jl_typemap_assoc_by_type(jl_atomic_load_relaxed(cache), &search, offs, /*subtype*/1);
        if (entry && jl_egal((jl_value_t*)entry->simplesig, simplett ? (jl_value_t*)simplett : jl_nothing) &&
                jl_egal((jl_value_t*)guardsigs, (jl_value_t*)entry->guardsigs)) {
            JL_GC_POP();
            return entry->func.linfo;
        }
    }

    jl_typemap_entry_t *newentry = jl_typemap_alloc(cachett, simplett, guardsigs, (jl_value_t*)newmeth, min_valid, max_valid);
    temp = (jl_value_t*)newentry;
    if (mc && cachett == tt && jl_svec_len(guardsigs) == 0 && tt->hash && !tt->hasfreetypevars) {
        // we check `tt->hash` exists, since otherwise the NamedTuple
        // constructor and `structdiff` method pollutes this lookup with a lot
        // of garbage in the linear table search
        if (jl_lookup_cache_type_(tt) == NULL) {
            // if this type isn't normally in the cache, force it in there now
            // anyways so that we can depend on it as a token (especially since
            // we just cached it in memory as this method signature anyways)
            JL_LOCK(&typecache_lock);
            if (jl_lookup_cache_type_(tt) == NULL)
                jl_cache_type_(tt);
            JL_UNLOCK(&typecache_lock); // Might GC
        }
        jl_genericmemory_t *oldcache = jl_atomic_load_relaxed(&mc->leafcache);
        jl_typemap_entry_t *old = (jl_typemap_entry_t*)jl_eqtable_get(oldcache, (jl_value_t*)tt, jl_nothing);
        jl_atomic_store_relaxed(&newentry->next, old);
        jl_gc_wb(newentry, old);
        jl_genericmemory_t *newcache = jl_eqtable_put(jl_atomic_load_relaxed(&mc->leafcache), (jl_value_t*)tt, (jl_value_t*)newentry, NULL);
        if (newcache != oldcache) {
            jl_atomic_store_release(&mc->leafcache, newcache);
            jl_gc_wb(mc, newcache);
        }
    }
    else {
         jl_typemap_insert(cache, parent, newentry, offs);
         if (mt) {
             jl_datatype_t *dt = jl_nth_argument_datatype((jl_value_t*)tt, 1);
             if (dt) {
                 jl_typename_t *tn = dt->name;
                 int cache_entry_count = jl_atomic_load_relaxed(&tn->cache_entry_count);
                 if (cache_entry_count < 31)
                     jl_atomic_store_relaxed(&tn->cache_entry_count, cache_entry_count + 1);
             }
         }
    }
    if (mc)
        jl_method_cache_inserted();
    if (mc) {
        JL_UNLOCK(&mc->writelock);

        // Only set METHOD_SIG_LATEST_ONLY on method instance if method does NOT have the bit, no guards required, and min_valid == primary_world
        int should_set_dispatch_status = !(jl_atomic_load_relaxed(&definition->dispatch_status) & METHOD_SIG_LATEST_ONLY) &&
            (!cache_with_orig && jl_svec_len(guardsigs) == 0) &&
            min_valid == jl_atomic_load_relaxed(&definition->primary_world) &&
            !(jl_atomic_load_relaxed(&newmeth->dispatch_status) & METHOD_SIG_LATEST_ONLY);

        // Combined trylock for both dispatch_status setting and max_world restoration
        if ((should_set_dispatch_status || unconstrained_max) &&
            jl_atomic_load_relaxed(&jl_world_counter) == current_world) {
            JL_LOCK(&world_counter_lock);
            if (jl_atomic_load_relaxed(&jl_world_counter) == current_world) {
                if (should_set_dispatch_status) {
                    jl_atomic_store_relaxed(&newmeth->dispatch_status, METHOD_SIG_LATEST_ONLY);
                }
                if (unconstrained_max) {
                    jl_atomic_store_relaxed(&newentry->max_world, ~(size_t)0);
                }
            }
            JL_UNLOCK(&world_counter_lock);
        }
    }

    JL_GC_POP();
    return newmeth;
}

JL_DLLEXPORT void jl_promote_cis_to_current(jl_code_instance_t **cis, size_t n, size_t validated_world)
{
    size_t current_world = jl_atomic_load_relaxed(&jl_world_counter);
    // No need to acquire the lock if we've been invalidated anyway
    if (current_world > validated_world)
        return;
    JL_LOCK(&world_counter_lock);
    current_world = jl_atomic_load_relaxed(&jl_world_counter);
    if (current_world == validated_world) {
        arraylist_t workqueue;
        arraylist_new(&workqueue, 0);
        for (size_t i = 0; i < n; i++)
            arraylist_push(&workqueue, cis[i]);
        while (workqueue.len > 0) {
            jl_code_instance_t *current_ci = (jl_code_instance_t *)arraylist_pop(&workqueue);
            if (jl_atomic_load_relaxed(&current_ci->max_world) != validated_world)
                continue;
            jl_atomic_store_relaxed(&current_ci->max_world, ~(size_t)0);
            jl_svec_t *edges = jl_atomic_load_relaxed(&current_ci->edges);
            for (size_t i = 0; i < jl_svec_len(edges); i++) {
                jl_value_t *edge = jl_svecref(edges, i);
                if (!jl_is_code_instance(edge))
                    continue;
                arraylist_push(&workqueue, edge);
            }
        }
        arraylist_free(&workqueue);
    }
    JL_UNLOCK(&world_counter_lock);
}

JL_DLLEXPORT void jl_promote_ci_to_current(jl_code_instance_t *ci, size_t validated_world)
{
    jl_promote_cis_to_current(&ci, 1, validated_world);
}

JL_DLLEXPORT void jl_promote_mi_to_current(jl_method_instance_t *mi, size_t min_world, size_t validated_world)
{
    size_t current_world = jl_atomic_load_relaxed(&jl_world_counter);
    // No need to acquire the lock if we've been invalidated anyway
    if (current_world > validated_world)
        return;
    // Only set METHOD_SIG_LATEST_ONLY on method instance if method does NOT have the bit and min_valid == primary_world
    jl_method_t *definition = mi->def.method;
    if ((jl_atomic_load_relaxed(&definition->dispatch_status) & METHOD_SIG_LATEST_ONLY) ||
        min_world != jl_atomic_load_relaxed(&definition->primary_world) ||
        (jl_atomic_load_relaxed(&mi->dispatch_status) & METHOD_SIG_LATEST_ONLY))
        return;
    JL_LOCK(&world_counter_lock);
    current_world = jl_atomic_load_relaxed(&jl_world_counter);
    if (current_world == validated_world) {
        jl_atomic_store_relaxed(&mi->dispatch_status, METHOD_SIG_LATEST_ONLY);
    }
    JL_UNLOCK(&world_counter_lock);
}

static jl_method_match_t *_gf_invoke_lookup(jl_value_t *types JL_PROPAGATES_ROOT, jl_methtable_t *mt, size_t world, int cache, size_t *min_valid, size_t *max_valid);

JL_DLLEXPORT jl_typemap_entry_t *jl_mt_find_cache_entry(jl_methcache_t *mc JL_PROPAGATES_ROOT, jl_datatype_t *tt JL_MAYBE_UNROOTED JL_ROOTS_TEMPORARILY, size_t world)
{ // exported only for debugging purposes, not for casual use
    if (tt->isdispatchtuple) {
        jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
        jl_typemap_entry_t *entry = lookup_leafcache(leafcache, (jl_value_t*)tt, world);
        if (entry)
            return entry;
    }
    JL_GC_PUSH1(&tt);
    struct jl_typemap_assoc search = {(jl_value_t*)tt, world, NULL};
    jl_typemap_entry_t *entry = jl_typemap_assoc_by_type(jl_atomic_load_relaxed(&mc->cache), &search, jl_cachearg_offset(), /*subtype*/1);
    JL_GC_POP();
    return entry;
}

static jl_method_instance_t *jl_mt_assoc_by_type(jl_methtable_t *mt, jl_methcache_t *mc JL_PROPAGATES_ROOT, jl_datatype_t *tt JL_MAYBE_UNROOTED, size_t world)
{
    size_t cache_insert_generation = jl_method_cache_insert_generation_load();
    jl_typemap_entry_t *entry = jl_mt_find_cache_entry(mc, tt, world);
    if (entry)
        return entry->func.linfo;
    assert(tt->isdispatchtuple || tt->hasfreetypevars);
    JL_TIMING(METHOD_LOOKUP_SLOW, METHOD_LOOKUP_SLOW);
    jl_method_match_t *matc = NULL;
    JL_GC_PUSH2(&tt, &matc);
    JL_LOCK(&mc->writelock);
    jl_method_instance_t *mi = NULL;
    // No need to re-check if no method cache insertion happened since the
    // lock-free probe above.
    int tt_known_absent = jl_method_cache_insert_generation_load() == cache_insert_generation;
    if (!tt_known_absent) {
        entry = jl_mt_find_cache_entry(mc, tt, world);
        if (entry)
            mi = entry->func.linfo;
    }
    if (!mi) {
        size_t min_valid = 0;
        size_t max_valid = ~(size_t)0;
        matc = _gf_invoke_lookup((jl_value_t*)tt, mt, world, 0, &min_valid, &max_valid);
        if (matc) {
            jl_method_t *m = matc->method;
            jl_svec_t *env = matc->sparams;
            // tt was just proven absent above and no method cache insertion has
            // happened since, so cache_method can skip its redundant re-check.
            mi = cache_method(mt, mc, &mc->cache, (jl_value_t*)mc, tt, m, world, min_valid, max_valid, env, tt_known_absent);
            JL_GC_POP();
            return mi;
        }
    }
    JL_UNLOCK(&mc->writelock);
    JL_GC_POP();
    return mi;
}

struct matches_env {
    struct typemap_intersection_env match;
    jl_typemap_entry_t *newentry;
    jl_value_t *shadowed;
    jl_typemap_entry_t *replaced;
    int replay; // §7: the "new entry" is a dummy standing for a membership, not an insertion
};

static int get_intersect_visitor(jl_typemap_entry_t *oldentry, struct typemap_intersection_env *closure0)
{
    struct matches_env *closure = container_of(closure0, struct matches_env, match);
    jl_method_t *oldmethod = oldentry->func.method;
    assert(oldentry != closure->newentry && "entry already added");
    if (closure->replay && jl_atomic_load_relaxed(&oldentry->max_world) != ~(size_t)0)
        return 1; // a replay asks about the last pre-membership world, which dead
                  // entries can still cover; their method cannot be selected in the
                  // new world either, and the asserts below assume a live insertion
    assert(jl_atomic_load_relaxed(&oldentry->min_world) <= jl_atomic_load_relaxed(&closure->newentry->min_world) && "old method cannot be newer than new method");
    //assert(jl_atomic_load_relaxed(&oldentry->max_world) != jl_atomic_load_relaxed(&closure->newentry->min_world) && "method cannot be added at the same time as method deleted");
    assert((jl_atomic_load_relaxed(&oldentry->max_world) == ~(size_t)0));
    // don't need to consider other similar methods if this oldentry will always fully intersect with them and dominates all of them
    if (!closure->replay && closure->match.issubty // e.g. jl_subtype(closure->newentry.sig, oldentry->sig)
        && jl_subtype(oldmethod->sig, (jl_value_t*)closure->newentry->sig)) { // e.g. jl_type_equal(closure->newentry->sig, oldentry->sig)
        if (closure->replaced == NULL || jl_atomic_load_relaxed(&closure->replaced->min_world) < jl_atomic_load_relaxed(&oldentry->min_world))
            closure->replaced = oldentry; // must pick the newest insertion (both are still valid)
    }
    if (closure->shadowed == NULL)
        closure->shadowed = (jl_value_t*)jl_alloc_vec_any(0);
    // This should be rarely true (in fact, get_intersect_visitor should be
    // rarely true), but might as well skip the rest of the scan fast anyways
    // since we can.
    if (closure->match.issubty) {
        int only = jl_atomic_load_relaxed(&oldmethod->dispatch_status) & METHOD_SIG_LATEST_ONLY;
        if (only && !closure->replay) {
            // replay (review M2a fix): the fast stop trusts LATEST_ONLY as
            // computed pre-membership; at the replay's world the entry was not
            // visible, and the walk must see the full intersect set — the
            // shadowed methods the entry would swallow here are exactly the
            // ones whose dispatch the declaration changes.
            size_t len = jl_array_nrows(closure->shadowed);
            if (len > 0)
                jl_array_del_end((jl_array_t*)closure->shadowed, len);
            jl_array_ptr_1d_push((jl_array_t*)closure->shadowed, (jl_value_t*)oldmethod);
            return 0;
        }
    }
    jl_array_ptr_1d_push((jl_array_t*)closure->shadowed, (jl_value_t*)oldmethod);
    typemap_slurp_search(oldentry, &closure->match);
    return 1;
}

static jl_value_t *get_intersect_matches(jl_typemap_t *defs, jl_typemap_entry_t *newentry, jl_typemap_entry_t **replaced, size_t world, int replay)
{
    jl_tupletype_t *type = newentry->sig;
    jl_tupletype_t *ttypes = (jl_tupletype_t*)jl_unwrap_unionall((jl_value_t*)type);
    size_t l = jl_nparams(ttypes);
    jl_value_t *va = NULL;
    if (l > 0) {
        va = jl_tparam(ttypes, l - 1);
        if (jl_is_vararg(va))
            va = jl_unwrap_vararg(va);
        else
            va = NULL;
    }
    // search for all intersecting methods active in the previous world, to determine the changes needed to be made for the next world
    struct matches_env env = {{get_intersect_visitor, (jl_value_t*)type, va, /* .search_slurp = */ 0,
            /* .world = */ world,
            /* .min_valid = */ world, /* .max_valid = */ world,
            /* .ti = */ NULL, /* .env = */ jl_emptysvec, /* .issubty = */ 0},
        /* .newentry = */ newentry, /* .shadowed */ NULL, /* .replaced */ NULL, /* .replay */ replay};
    // the shadowed and replaced methods are the answer, not the matches' worlds: no membership range is
    // taken (at the world before a membership the walk asks at, it would walk every entry, E-303)
    env.match.skip_ranges = 1;
    JL_GC_PUSH3(&env.match.env, &env.match.ti, &env.shadowed);
    jl_typemap_intersection_visitor(defs, 0, &env.match);
    env.match.env = NULL;
    env.match.ti = NULL;
    *replaced = env.replaced;
    JL_GC_POP();
    return env.shadowed;
}

void print_func_loc(JL_STREAM *s, jl_method_t *m)
{
    long lno = m->line;
    if (lno > 0) {
        char *fname = jl_symbol_name((jl_sym_t*)m->file);
        jl_printf(s, " at %s:%ld", fname, lno);
    }
}

static void method_overwrite(jl_typemap_entry_t *newentry, jl_method_t *oldvalue)
{
    // method overwritten
    jl_method_t *method = (jl_method_t*)newentry->func.method;
    jl_module_t *newmod = method->module;
    jl_module_t *oldmod = oldvalue->module;
    jl_datatype_t *dt = jl_nth_argument_datatype(oldvalue->sig, 1);
    if (jl_kwcall_type && dt == jl_kwcall_type)
        dt = jl_nth_argument_datatype(oldvalue->sig, 3);
    int anon = dt && is_anonfn_typename(jl_symbol_name(dt->name->name));
    if ((jl_options.warn_overwrite == JL_OPTIONS_WARN_OVERWRITE_ON) ||
        (jl_options.incremental && jl_generating_output()) || anon) {
        JL_STREAM *s = JL_STDERR;
        jl_printf(s, "WARNING: Method definition ");
        jl_static_show_func_sig(s, (jl_value_t*)newentry->sig);
        jl_printf(s, " in module %s", jl_symbol_name(oldmod->name));
        print_func_loc(s, oldvalue);
        jl_printf(s, " overwritten");
        if (oldmod != newmod)
            jl_printf(s, " in module %s", jl_symbol_name(newmod->name));
        if (method->line > 0 && method->line == oldvalue->line && method->file == oldvalue->file)
            jl_printf(s, anon ? " on the same line" : " on the same line (check for duplicate calls to `include`)");
        else
            print_func_loc(s, method);
        jl_printf(s, ".\n");
        jl_uv_flush(s);
    }
    if (jl_generating_output() && jl_options.incremental) {
        jl_printf(JL_STDERR, "ERROR: Method overwriting is not permitted during Module precompilation. Use `__precompile__(false)` to opt-out of precompilation.\n");
        jl_throw(jl_precompilable_error);
    }
}

static void update_max_args(jl_value_t *type)
{
    type = jl_unwrap_unionall(type);
    jl_datatype_t *dt = jl_nth_argument_datatype(type, 1);
    if (dt == NULL || dt == jl_kwcall_type || jl_is_type_type((jl_value_t*)dt))
        return;
    jl_typename_t *tn = dt->name;
    assert(jl_is_datatype(type));
    size_t na = jl_nparams(type);
    if (jl_va_tuple_kind((jl_datatype_t*)type) == JL_VARARG_UNBOUND)
        na--;
    // update occurs inside global writelock
    if (na > jl_atomic_load_relaxed(&tn->max_args))
        jl_atomic_store_relaxed(&tn->max_args, na);
}

// the set->methods index (plan §4.2, review M10): the methods whose signatures name
// an open set, so the declaration-time walk (§7) and the ambiguity report (§8) can
// find them. rare (E-253: 131 across four ports), so a globally rooted flat vec of
// (set typename, method) pairs with a linear scan is the whole design; pairs are
// never removed, a stale one is filtered at use.
// Review A-F1 (BLOCKER): rooted for real, not just for the analyzer —
// JL_GLOBALLY_ROOTED is an annotation (support/analyzer_annotations.h); the stock
// collector marks C globals explicitly in gc_mark_roots and this vec follows the
// _jl_debug_method_invalidation pattern (non-static global, extern in
// julia_internal.h, mark + heap-snapshot lines in gc-stock.c / gc-mmtk.c). Two
// full GC sweeps used to sweep the vec itself, and the next push wrote a freed
// array (m2r2A-b3/segv3 dumped core).
jl_array_t *open_set_methods JL_GLOBALLY_ROOTED = NULL;
static jl_mutex_t open_set_methods_lock; // zero-init, like world_counter_lock

static void open_set_walk_names(jl_value_t *v, jl_method_t *m, int depth)
{
    // structural: register m under every open-set typename inside the sig
    // (direct argument positions today; nested occurrences are indexed too, so
    // the §7 walk can see them and choose its conservative path); depth caps at 8.
    // review W2-1 (BLOCKER): a Vararg node is not a datatype — without this arm a
    // `Tuple{typeof(f), Vararg{OpenSet}}` method indexed NOTHING, §7 never saw
    // it, and stale dispatch persisted in the NEW world. Recurse into the
    // element exactly like set_name_nested does.
    v = jl_unwrap_unionall(v);
    if (depth > 8)
        return;
    if (jl_is_vararg(v)) {
        open_set_walk_names(jl_unwrap_vararg(v), m, depth + 1);
        return;
    }
    if (jl_is_uniontype(v)) {
        // review A (m2_future_min_worlds (A)): a set under a UNION names the set
        // too — without this arm `Tuple{typeof(f), Union{Other, OpenSet}}`
        // indexed nothing (Union is not a datatype), §7 never saw the method,
        // and a caller compiled pre-declare kept its stale answer. Same class
        // as the Vararg gap above; same fix shape as set_name_nested.
        open_set_walk_names(((jl_uniontype_t*)v)->a, m, depth + 1);
        open_set_walk_names(((jl_uniontype_t*)v)->b, m, depth + 1);
        return;
    }
    if (!jl_is_datatype(v))
        return;
    jl_datatype_t *dt = (jl_datatype_t*)v;
    if (dt->name->open_set) {
        JL_LOCK(&open_set_methods_lock);
        if (!open_set_methods)
            open_set_methods = jl_alloc_vec_any(0);
        jl_array_ptr_1d_push(open_set_methods, (jl_value_t*)dt->name);
        jl_array_ptr_1d_push(open_set_methods, (jl_value_t*)m);
        JL_UNLOCK(&open_set_methods_lock);
    }
    for (size_t i = 0; i < jl_nparams(dt); i++)
        open_set_walk_names(jl_tparam(dt, i), m, depth + 1);
}

// called from jl_method_table_add. this is the only writer, and it runs before
// insert takes world_counter_lock, so the lock stays the innermost nest (W4-10).
// Each function's methods at an argument typed by a set (compiler §5.6): the function's TypeName to a
// list of methods (a SimpleVector, replaced). The shadow query of a call reads its function's list.
jl_genericmemory_t *jl_func_set_methods JL_GLOBALLY_ROOTED = NULL;

static void index_func_set_method(jl_method_t *m)
{
    jl_value_t *sig = jl_unwrap_unionall(m->sig);
    if (!jl_is_datatype(sig) || jl_nparams(sig) == 0)
        return;
    jl_value_t *ft = jl_unwrap_unionall(jl_tparam0(sig));
    if (!jl_is_datatype(ft))
        return;
    jl_typename_t *fname = ((jl_datatype_t*)ft)->name;
    jl_svec_t *nw = NULL;
    JL_GC_PUSH1(&nw);
    jl_svec_t *old = jl_func_set_methods ? (jl_svec_t*)jl_eqtable_get(jl_func_set_methods, (jl_value_t*)fname, NULL) : NULL;
    size_t n = old ? jl_svec_len(old) : 0;
    nw = jl_alloc_svec_uninit(n + 1);
    for (size_t i = 0; i < n; i++)
        jl_svecset(nw, i, jl_svecref(old, i));
    jl_svecset(nw, n, (jl_value_t*)m);
    if (jl_func_set_methods == NULL)
        jl_func_set_methods = jl_alloc_memory_any(16);
    jl_func_set_methods = jl_eqtable_put(jl_func_set_methods, (jl_value_t*)fname, (jl_value_t*)nw, NULL);
    JL_GC_POP();
}

static void jl_index_open_set_methods(jl_method_t *m)
{
    if (!jl_has_open_set_in((jl_value_t*)m->sig)) // one bit; false for every Base method (I3)
        return;
    open_set_walk_names((jl_value_t*)m->sig, m, 0);
}

// The methods whose signatures name the set (its TypeName, or a type of it): the set-to-methods
// index (compiler §5.1), for compiling ahead (§9; traits §16). Each method once, in index order.
JL_DLLEXPORT jl_value_t *jl_open_set_methods_of(jl_value_t *set)
{
    jl_value_t *uw = jl_unwrap_unionall(set);
    jl_typename_t *tn = jl_is_typename(uw) ? (jl_typename_t*)uw :
                        jl_is_datatype(uw) ? ((jl_datatype_t*)uw)->name : NULL;
    jl_array_t *out = jl_alloc_vec_any(0);
    if (tn == NULL)
        return (jl_value_t*)out;
    JL_GC_PUSH1(&out);
    JL_LOCK(&open_set_methods_lock);
    size_t n = open_set_methods ? jl_array_nrows(open_set_methods) : 0;
    for (size_t i = 0; i + 1 < n; i += 2) {
        if (jl_array_ptr_ref(open_set_methods, i) != (jl_value_t*)tn)
            continue;
        jl_value_t *m = jl_array_ptr_ref(open_set_methods, i + 1);
        int dup = 0;
        for (size_t k = 0; k < jl_array_nrows(out); k++)
            dup |= jl_array_ptr_ref(out, k) == m;
        if (!dup)
            jl_array_ptr_1d_push(out, m);
    }
    JL_UNLOCK(&open_set_methods_lock);
    JL_GC_POP();
    return (jl_value_t*)out;
}

static int is_set_type(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (jl_is_typevar(t))
        t = ((jl_tvar_t*)t)->ub;
    t = jl_unwrap_unionall(t);
    return jl_is_datatype(t) && (((jl_datatype_t*)t)->name->open_set || jl_is_all_type(t));
}

// whether an argument of type `a` could meet the declared type `d`: now, or by a later membership
static int could_meet(jl_value_t *a, jl_value_t *d)
{
    if (jl_is_typevar(d))
        d = ((jl_tvar_t*)d)->ub;
    if (!jl_has_empty_intersection(a, d))
        return 1;
    return is_set_type(d) && jl_could_join(a, d);
}

// the declared type of the method signature `sig` at argument `i`: a varargs tail's element there
static jl_value_t *sig_arg(jl_value_t *sig, size_t i, int isva) JL_NOTSAFEPOINT
{
    size_t n = jl_nparams(sig);
    if (n == 0)
        return NULL;
    if (i < n - 1 || (i == n - 1 && !jl_is_vararg(jl_tparam(sig, n - 1))))
        return jl_tparam(sig, i);
    jl_value_t *last = jl_tparam(sig, n - 1);
    return jl_is_vararg(last) ? jl_unwrap_vararg(last) : NULL;
}

// the sets named in a declared type: itself, a union's arms, an intersection's components, parameters
static void set_names_in(jl_value_t *t, arraylist_t *out, int depth) JL_NOTSAFEPOINT
{
    if (t == NULL || depth > 8)
        return;
    if (jl_is_typevar(t)) {
        set_names_in(((jl_tvar_t*)t)->ub, out, depth + 1);
        return;
    }
    t = jl_unwrap_unionall(t);
    if (jl_is_vararg(t)) {
        set_names_in(jl_unwrap_vararg(t), out, depth + 1);
        return;
    }
    if (jl_is_uniontype(t)) {
        set_names_in(((jl_uniontype_t*)t)->a, out, depth + 1);
        set_names_in(((jl_uniontype_t*)t)->b, out, depth + 1);
        return;
    }
    if (!jl_is_datatype(t) || !((jl_datatype_t*)t)->has_open_set)
        return;
    jl_typename_t *tn = ((jl_datatype_t*)t)->name;
    if (tn->open_set) {
        size_t k;
        for (k = 0; k < out->len; k++)
            if (out->items[k] == tn)
                break;
        if (k == out->len)
            arraylist_push(out, tn);
    }
    for (size_t k = 0; k < jl_nparams(t); k++)
        set_names_in(jl_tparam(t, k), out, depth + 1);
}

// Whether the call with argument types `tt`, which selects `m0` (or none), has a shadow at argument
// `i` by the method `mz` (compiler §5.6): `mz` declares there a type that names a set (the set, a
// union's arm, an intersection, a parameter), which the argument's concrete type does not meet now
// but could by a later membership; `m0` takes the argument as `Any` or at a type naming a set (a
// method matched through the type's own supertypes is more specific than one at a set); `mz`'s other
// arguments could meet the call's.
// A method that can only raise at the members of the sets it is at (compiler §5.6, §6.8): its declared
// result is `Union{}`, or it serves no member (`declared_covers` bit 1, a body's `serves_no_member` meta)
static int method_raises_only(jl_method_t *m) JL_NOTSAFEPOINT
{
    return m->declared_result == (jl_value_t*)jl_bottom_type || (m->declared_covers & 2);
}

static int shadow_at(jl_value_t *tt, jl_method_t *m0, jl_method_t *mz, size_t i)
{
    // open issue 158 (a): a method that can only raise at a member is no shadow: dispatched, it
    // throws as the pruned arm would
    if (method_raises_only(mz))
        return 0;
    jl_value_t *zsig = jl_unwrap_unionall(mz->sig);
    if (!jl_is_datatype(zsig) || !jl_is_datatype(tt))
        return 0;
    size_t np = jl_nparams(tt), nz = jl_nparams(zsig);
    if (!(mz->isva ? np >= nz - 1 : np == nz) || i >= np)
        return 0;
    jl_value_t *d = sig_arg(zsig, i, mz->isva);
    if (d == NULL || !jl_has_open_set_in(d))
        return 0;
    jl_value_t *s = jl_tparam(tt, i);
    if (jl_is_vararg(s) || !jl_is_concrete_type(s) || ((jl_datatype_t*)s)->name->never_member)
        return 0;
    jl_value_t *db = jl_is_typevar(d) ? ((jl_tvar_t*)d)->ub : d;
    if (jl_subtype(s, db) || !jl_could_join(s, db))
        return 0;
    jl_value_t *m0sig = m0 ? jl_unwrap_unionall(m0->sig) : NULL;
    if (m0sig != NULL && jl_is_datatype(m0sig)) {
        jl_value_t *d0 = sig_arg(m0sig, i, m0->isva);
        if (d0 != NULL && jl_is_typevar(d0))
            d0 = ((jl_tvar_t*)d0)->ub;
        if (d0 != NULL && d0 != (jl_value_t*)jl_any_type && !jl_has_open_set_in(d0))
            return 0; // matched through the type's own supertypes
    }
    // the other arguments, the function's own type included: the methods at sets indexed under `Type`
    // are every type's constructors, whose first argument (`Type{X}`) tells them apart
    for (size_t j = 0; j < np; j++) {
        jl_value_t *dj = sig_arg(zsig, j, mz->isva);
        if (j == i || dj == NULL || jl_is_vararg(jl_tparam(tt, j)))
            continue;
        if (!could_meet(jl_tparam(tt, j), dj))
            return 0;
    }
    return 1;
}

// The shadows of a call (compiler §5.6): the call has argument types `tt` and selects `m0` (or none).
// A shadow is an argument whose concrete type could later join a set at which another method of the
// function could then be selected, where `m0` takes that argument as `Any` or at a set (a method
// matched through the type's own supertypes is more specific than any method at a set). Returned
// as svec(cells, results): the cell of each (argument type, set), which the call's guard reads, and
// each shadowing method's declared result, `Any` where it declares none. Empty where there is none.
JL_DLLEXPORT jl_value_t *jl_call_shadows(jl_value_t *tt, jl_value_t *m0v)
{
    if (!jl_is_datatype(tt) || !jl_is_tuple_type(tt) || jl_nparams(tt) < 2 || jl_func_set_methods == NULL)
        return (jl_value_t*)jl_emptysvec;
    jl_value_t *ft = jl_unwrap_unionall(jl_tparam0(tt));
    if (!jl_is_datatype(ft))
        return (jl_value_t*)jl_emptysvec;
    jl_svec_t *ms = NULL;
    JL_LOCK(&open_set_methods_lock);
    ms = (jl_svec_t*)jl_eqtable_get(jl_func_set_methods, (jl_value_t*)((jl_datatype_t*)ft)->name, NULL);
    JL_UNLOCK(&open_set_methods_lock);
    if (ms == NULL)
        return (jl_value_t*)jl_emptysvec;
    jl_method_t *m0 = jl_is_method(m0v) ? (jl_method_t*)m0v : NULL;
    size_t np = jl_nparams(tt);
    jl_array_t *cells = NULL, *results = NULL;
    jl_value_t *cell = NULL, *r = NULL;
    JL_GC_PUSH5(&ms, &cells, &results, &cell, &r);
    for (size_t k = 0; k < jl_svec_len(ms); k++) {
        jl_method_t *mz = (jl_method_t*)jl_svecref(ms, k);
        if (mz == m0)
            continue;
        jl_value_t *zsig = jl_unwrap_unionall(mz->sig);
        if (!jl_is_datatype(zsig))
            continue;
        for (size_t i = 1; i < np; i++) {
            if (!shadow_at(tt, m0, mz, i))
                continue;
            jl_value_t *s = jl_tparam(tt, i);
            if (cells == NULL) {
                cells = jl_alloc_vec_any(0);
                results = jl_alloc_vec_any(0);
            }
            // a cell for each set the declared type names: a membership in any of them opens the guard
            arraylist_t names;
            arraylist_new(&names, 0);
            set_names_in(sig_arg(zsig, i, mz->isva), &names, 0);
            for (size_t k = 0; k < names.len; k++) {
                cell = jl_membership_cell(s, (jl_value_t*)names.items[k]);
                jl_array_ptr_1d_push(cells, cell);
                r = mz->declared_result ? mz->declared_result : (jl_value_t*)jl_any_type;
                // a declared result that names the method's static parameters (`Tuple{T, Any}` at
                // `Iterable{T}`) is not instantiated at the call: the shadow's result is `Any`, never a
                // type with a free variable in the caller's result
                if (jl_has_free_typevars(r))
                    r = (jl_value_t*)jl_any_type;
                jl_array_ptr_1d_push(results, r);
            }
            arraylist_free(&names);
        }
    }
    jl_value_t *out = (jl_value_t*)jl_emptysvec;
    if (cells != NULL) {
        size_t n = jl_array_nrows(cells);
        jl_svec_t *cs = jl_alloc_svec(n), *rs = NULL;
        JL_GC_PUSH2(&cs, &rs);
        rs = jl_alloc_svec(n);
        for (size_t i = 0; i < n; i++) {
            jl_svecset(cs, i, jl_array_ptr_ref(cells, i));
            jl_svecset(rs, i, jl_array_ptr_ref(results, i));
        }
        out = (jl_value_t*)jl_svec2(cs, rs);
        JL_GC_POP();
    }
    JL_GC_POP();
    return out;
}

// used by §7/§8: copy out the methods indexed under `tn`
static jl_array_t *jl_methods_at_set(jl_typename_t *tn)
{
    jl_array_t *out = jl_alloc_vec_any(0);
    JL_GC_PUSH1(&out);
    JL_LOCK(&open_set_methods_lock);
    if (open_set_methods) {
        size_t i, l = jl_array_nrows(open_set_methods);
        for (i = 0; i + 1 < l; i += 2)
            if (jl_array_ptr_ref(open_set_methods, i) == (jl_value_t*)tn)
                jl_array_ptr_1d_push(out, jl_array_ptr_ref(open_set_methods, i + 1));
    }
    JL_UNLOCK(&open_set_methods_lock);
    JL_GC_POP();
    return out;
}

// The table of a call's targets (compiler §7.3; Compiler/src/splittable.jl, `SplitTable`): one table
// per call signature, from the value's type to a `:caller_world` pointer to the call's target
// compiled for that type. The layouts mirror the Julia types. A table's state is replaced on each
// addition, never mutated, so a reader loads it once; a state of another session (a table saved in
// a package image) counts as empty.
typedef struct {
    jl_genericmemory_t *kv;   // Memory{UInt}: pairs (type address, pointer) by open addressing; 0 is empty
    uint64_t session;         // the session that filled it
    jl_genericmemory_t *keys; // Memory{Any}: the types, rooted
} jl_split_state_t;
typedef struct {
    _Atomic(jl_split_state_t*) state;
} jl_split_table_t;

static _Atomic(uint64_t) jl_split_session_id = 0;
static jl_value_t *jl_split_table_miss_func JL_GLOBALLY_ROOTED = NULL;

// This session's stamp for split tables: made at its first use, never 0.
JL_DLLEXPORT uint64_t jl_split_session(void)
{
    uint64_t s = jl_atomic_load_relaxed(&jl_split_session_id);
    if (s == 0) {
        uint64_t n = jl_hrtime() | 1;
        if (jl_atomic_cmpswap(&jl_split_session_id, &s, n))
            s = n;
    }
    return s;
}

// Base's `_split_table_miss!(tbl, T)`: makes the pointer for a type met the first time, and adds it.
JL_DLLEXPORT void jl_set_split_table_miss(jl_value_t *f)
{
    jl_split_table_miss_func = f;
}

// The pointer for the type `T` in split table `tbl`: one probe, else made by the miss function, in
// the caller's world.
JL_DLLEXPORT void *jl_split_table_ptr(jl_value_t *tbl, jl_value_t *T)
{
    jl_split_state_t *st = jl_atomic_load_acquire(&((jl_split_table_t*)tbl)->state);
    if (st->session == jl_atomic_load_relaxed(&jl_split_session_id)) {
        jl_genericmemory_t *kv = st->kv;
        uint64_t *d = (uint64_t*)kv->ptr;
        size_t mask = kv->length / 2 - 1;
        uint64_t a = (uint64_t)(uintptr_t)T;
        size_t i = (size_t)(a >> 4) & mask;
        while (1) {
            uint64_t k = d[2 * i];
            if (k == a)
                return (void*)(uintptr_t)d[2 * i + 1];
            if (k == 0)
                break;
            i = (i + 1) & mask;
        }
    }
    if (jl_split_table_miss_func == NULL)
        jl_error("a split's table was reached before Base was initialized");
    jl_value_t *args[2] = {tbl, T};
    jl_value_t *fp = jl_apply_generic(jl_split_table_miss_func, args, 2);
    return jl_unbox_voidpointer(fp);
}

// G2 as a predicate (compiler §3.2): set once a CodeInstance's record is finished and satisfies it.
JL_DLLEXPORT void jl_ci_set_g2(jl_code_instance_t *ci)
{
    jl_atomic_fetch_or_relaxed(&ci->flags, JL_CI_FLAGS_G2);
}

// A method's declared result (compiler §6.8). The library sets it on a trait's method for an
// operation, with `covers` set: each member's serving method is checked to infer within it.
JL_DLLEXPORT void jl_method_set_declared_result(jl_method_t *m, jl_value_t *rt, int covers)
{
    if (!jl_is_type(rt) && !jl_is_typevar(rt))
        jl_type_error("declared result", (jl_value_t*)jl_type_type, rt);
    m->declared_result = rt;
    jl_gc_wb(m, rt);
    m->declared_covers = (m->declared_covers & ~1) | (covers != 0);
}

// The methods at open sets of the function type `ft` that carry a declared result covering
// every member: what inference reads at a receiver typed by a set (compiler §7.1). A method
// appears once per set its signature names.
JL_DLLEXPORT jl_value_t *jl_declared_methods_of(jl_value_t *ft)
{
    jl_array_t *out = jl_alloc_vec_any(0);
    JL_GC_PUSH1(&out);
    JL_LOCK(&open_set_methods_lock);
    if (open_set_methods) {
        size_t i, l = jl_array_nrows(open_set_methods);
        for (i = 0; i + 1 < l; i += 2) {
            jl_method_t *m = (jl_method_t*)jl_array_ptr_ref(open_set_methods, i + 1);
            if (!(m->declared_covers & 1) || m->declared_result == NULL)
                continue;
            jl_value_t *sig = jl_unwrap_unionall(m->sig);
            if (!jl_is_tuple_type(sig) || jl_nparams(sig) == 0 || !jl_egal(jl_tparam0(sig), ft))
                continue;
            jl_array_ptr_1d_push(out, (jl_value_t*)m);
        }
    }
    JL_UNLOCK(&open_set_methods_lock);
    JL_GC_POP();
    return (jl_value_t*)out;
}

// the walk's counters (plan §5.M2's log gate): what one declaration cost
struct membership_stats {
    size_t declares;
    size_t replays;            // methods re-resolved as if re-inserted
    size_t visits;             // intersecting methods examined
    size_t morespec_skips;     // of those, skipped as more specific (C7's B << A)
    size_t spec_invalidations; // dispatch-replacement EVENTS (review B F-B3): a
                               // replay stamps every MI of every replaced spec it
                               // reaches, so one declaration counts replays x MIs,
                               // not distinct MethodInstances
    size_t cache_lowerings;    // typemap entries whose max_world was lowered
    size_t latest_only_clears; // METHOD_SIG_LATEST_ONLY fast paths cleared
};
static struct membership_stats membership_stats;

// gates read these by index, in the order of the struct's fields
JL_DLLEXPORT size_t jl_membership_stat(int which)
{
    switch (which) {
    case 0: return membership_stats.declares;
    case 1: return membership_stats.replays;
    case 2: return membership_stats.visits;
    case 3: return membership_stats.morespec_skips;
    case 4: return membership_stats.spec_invalidations;
    case 5: return membership_stats.cache_lowerings;
    case 6: return membership_stats.latest_only_clears;
    default: return 0;
    }
}

static int membership_log_on = -1;

// one line per declare on JULIA_MEMBERSHIP_LOG, method_overwrite's style
static void jl_membership_log(jl_value_t *member, jl_value_t *set_instance, size_t max_world)
{
    if (membership_log_on == -1)
        membership_log_on = getenv("JULIA_MEMBERSHIP_LOG") != NULL;
    if (!membership_log_on)
        return;
    jl_printf(JL_STDERR, "membership ");
    jl_static_show(JL_STDERR, member);
    jl_printf(JL_STDERR, " in ");
    jl_static_show(JL_STDERR, set_instance);
    jl_printf(JL_STDERR, ": replays=%zu visits=%zu morespec_skips=%zu specs=%zu caches=%zu latest_only=%zu max_world=%zu\n",
              membership_stats.replays, membership_stats.visits,
              membership_stats.morespec_skips, membership_stats.spec_invalidations,
              membership_stats.cache_lowerings, membership_stats.latest_only_clears,
              max_world);
    jl_uv_flush(JL_STDERR);
}

jl_array_t *_jl_debug_method_invalidation JL_GLOBALLY_ROOTED = NULL;
JL_DLLEXPORT jl_value_t *jl_debug_method_invalidation(int state)
{
    /* After calling with `state = 1`, caller is responsible for
       holding a reference to the returned array until this is called
       again with `state = 0`. */
    if (state) {
        if (_jl_debug_method_invalidation)
            return (jl_value_t*) _jl_debug_method_invalidation;
        _jl_debug_method_invalidation = jl_alloc_array_1d(jl_array_any_type, 0);
        return (jl_value_t*) _jl_debug_method_invalidation;
    }
    _jl_debug_method_invalidation = NULL;
    return jl_nothing;
}

static void _invalidate_backedges(jl_method_instance_t *replaced_mi, jl_code_instance_t *replaced_ci, size_t max_world, int depth);

// The code slot (compiler §3.3): its switch, its counts, and the record of result edges.
static struct {
    _Atomic(size_t) marks;       // callers marked for revalidation
    _Atomic(size_t) kept;        // marked callers made valid again
    _Atomic(size_t) final_marks; // marks that became invalidations
    _Atomic(size_t) misses;      // slot misses resolved by the runtime
    _Atomic(size_t) fallbacks;   // misses served by the dispatching last resort
} code_slot_stats;
#define CODE_SLOT_COUNT(f) jl_atomic_fetch_add_relaxed(&code_slot_stats.f, 1)

static int code_slot_on = -1;
JL_DLLEXPORT int jl_code_slot_on(void) JL_NOTSAFEPOINT
{
    if (code_slot_on == -1) {
        const char *e = getenv("JULIA_CODE_SLOT");
        code_slot_on = !(e && strcmp(e, "0") == 0);
    }
    return code_slot_on;
}

JL_DLLEXPORT size_t jl_code_slot_stat(int which) JL_NOTSAFEPOINT
{
    switch (which) {
    case 0: return jl_atomic_load_relaxed(&code_slot_stats.marks);
    case 1: return jl_atomic_load_relaxed(&code_slot_stats.kept);
    case 2: return jl_atomic_load_relaxed(&code_slot_stats.final_marks);
    case 3: return jl_atomic_load_relaxed(&code_slot_stats.misses);
    case 4: return jl_atomic_load_relaxed(&code_slot_stats.fallbacks);
    }
    return 0;
}

// the field `k` of a struct whose fields are references, or NULL
static jl_value_t *ptr_field(jl_value_t *v, size_t k) JL_NOTSAFEPOINT
{
    jl_datatype_t *t = (jl_datatype_t*)jl_typeof(v);
    if (!jl_is_datatype(t) || t->layout == NULL || k >= jl_datatype_nfields(t) || !jl_field_isptr(t, k))
        return NULL;
    return *(jl_value_t**)((char*)v + jl_field_offset(t, k));
}

// a struct of the compiler's (module `Compiler`), by its name and number of fields
static int is_named_struct(jl_value_t *v, const char *name, size_t nfields) JL_NOTSAFEPOINT
{
    jl_datatype_t *t = (jl_datatype_t*)jl_typeof(v);
    return jl_is_datatype(t) && jl_datatype_nfields(t) == nfields &&
        strcmp(jl_symbol_name(t->name->name), name) == 0 &&
        strcmp(jl_symbol_name(t->name->module->name), "Compiler") == 0;
}

// the compiler's `ResultEdges` on the CodeInstance's analysis results: the MethodInstances it reaches
// only by `:invoke`s typed by their results, which its code calls through slots
static jl_svec_t *ci_result_edges(jl_code_instance_t *ci) JL_NOTSAFEPOINT
{
    jl_value_t *ar = ci->analysis_results;
    while (ar != NULL && is_named_struct(ar, "AnalysisResults", 2)) {
        jl_value_t *r = ptr_field(ar, 0);
        if (r != NULL && is_named_struct(r, "ResultEdges", 1)) {
            jl_value_t *mis = ptr_field(r, 0);
            return mis != NULL && jl_is_svec(mis) ? (jl_svec_t*)mis : NULL;
        }
        ar = ptr_field(ar, 1);
    }
    return NULL;
}

JL_DLLEXPORT int jl_ci_has_result_edge(jl_code_instance_t *ci, jl_method_instance_t *mi) JL_NOTSAFEPOINT
{
    jl_svec_t *mis = ci_result_edges(ci);
    if (mis == NULL)
        return 0;
    for (size_t i = 0; i < jl_svec_len(mis); i++)
        if (jl_svecref(mis, i) == (jl_value_t*)mi)
            return 1;
    return 0;
}

// recursively invalidate cached methods that had an edge to a replaced method
static void invalidate_code_instance(jl_code_instance_t *replaced, size_t max_world, int depth)
{
    jl_timing_counter_inc(JL_TIMING_COUNTER_Invalidations, 1);
    if (_jl_debug_method_invalidation) {
        jl_value_t *boxeddepth = NULL;
        JL_GC_PUSH1(&boxeddepth);
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)replaced->def);
        boxeddepth = jl_box_int32(depth);
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, boxeddepth);
        JL_GC_POP();
    }
    //jl_static_show(JL_STDERR, (jl_value_t*)replaced->def);
    jl_method_instance_t *replaced_mi = jl_get_ci_mi(replaced);
    if (!jl_is_method(replaced_mi->def.method))
        return; // shouldn't happen, but better to be safe
    JL_LOCK(&replaced_mi->def.method->writelock);
    size_t replacedmaxworld = jl_atomic_load_relaxed(&replaced->max_world);
    // an invalidation of a marked CodeInstance makes the mark final (compiler §3.3): its callers
    // were walked when it was marked
    if (jl_atomic_load_relaxed(&replaced->flags) & JL_CI_FLAGS_MARKED) {
        jl_atomic_fetch_and_relaxed(&replaced->flags, (uint8_t)~JL_CI_FLAGS_MARKED);
        CODE_SLOT_COUNT(final_marks);
    }
    if (replacedmaxworld == ~(size_t)0) {
        assert(jl_atomic_load_relaxed(&replaced->min_world) - 1 <= max_world && "attempting to set illogical world constraints (probable race condition)");
        jl_atomic_store_release(&replaced->max_world, max_world);
        // recurse to all backedges to update their valid range also
        _invalidate_backedges(replaced_mi, replaced, max_world, depth + 1);
        // TODO: should we visit all forward edges now and delete ourself from all of those lists too?
    }
    else {
        assert(jl_atomic_load_relaxed(&replaced->max_world) <= max_world);
    }
    JL_UNLOCK(&replaced_mi->def.method->writelock);
}

// Marks a caller for revalidation instead of invalidating it (compiler §3.3): its max_world is
// lowered as an invalidation's, so every cache that checks worlds misses it, and the flag says that
// its next lookup in the latest world may compare its result edges and make it valid again. The walk
// goes on from it as from an invalidated one. Logged as an invalidation.
static void mark_code_instance(jl_code_instance_t *ci, size_t max_world, int depth)
{
    jl_timing_counter_inc(JL_TIMING_COUNTER_Invalidations, 1);
    if (_jl_debug_method_invalidation) {
        jl_value_t *boxeddepth = NULL;
        JL_GC_PUSH1(&boxeddepth);
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)ci->def);
        boxeddepth = jl_box_int32(depth);
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, boxeddepth);
        JL_GC_POP();
    }
    jl_method_instance_t *mi = jl_get_ci_mi(ci);
    if (!jl_is_method(mi->def.method))
        return;
    JL_LOCK(&mi->def.method->writelock);
    if (jl_atomic_load_relaxed(&ci->max_world) == ~(size_t)0) {
        jl_atomic_fetch_or_relaxed(&ci->flags, JL_CI_FLAGS_MARKED);
        jl_atomic_store_release(&ci->max_world, max_world);
        CODE_SLOT_COUNT(marks);
        _invalidate_backedges(mi, ci, max_world, depth + 1);
    }
    JL_UNLOCK(&mi->def.method->writelock);
}

JL_DLLEXPORT void jl_invalidate_code_instance(jl_code_instance_t *replaced, size_t max_world)
{
    invalidate_code_instance(replaced, max_world, 1);
}

JL_DLLEXPORT void jl_maybe_log_binding_invalidation(jl_value_t *replaced)
{
    if (_jl_debug_method_invalidation) {
        if (replaced) {
            jl_array_ptr_1d_push(_jl_debug_method_invalidation, replaced);
        }
        jl_value_t *loctag = jl_cstr_to_string("jl_maybe_log_binding_invalidation");
        JL_GC_PUSH1(&loctag);
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, loctag);
        JL_GC_POP();
    }
}

static void _invalidate_backedges(jl_method_instance_t *replaced_mi, jl_code_instance_t *replaced_ci, size_t max_world, int depth) {
    uint8_t recursion_flags = 0;
    jl_array_t *backedges = jl_mi_get_backedges_mutate(replaced_mi, &recursion_flags);
    if (!backedges)
        return;
    // invalidate callers (if any)
    if (!replaced_ci) {
        // We know all backedges are deleted - clear them eagerly
        // Clears both array and flags
        replaced_mi->backedges = NULL;
        jl_atomic_fetch_and_relaxed(&replaced_mi->flags, ~MI_FLAG_BACKEDGES_ALL);
    }
    JL_GC_PUSH1(&backedges);
    size_t i = 0, l = jl_array_nrows(backedges);
    size_t ins = 0;
    jl_code_instance_t *replaced;
    while (i < l) {
        jl_value_t *invokesig = NULL;
        i = get_next_edge(backedges, i, &invokesig, &replaced);
        if (!replaced) {
            ins = i;
            continue;
        }
        JL_GC_PROMISE_ROOTED(replaced); // propagated by get_next_edge from backedges
        if (replaced_ci) {
            // If we're invalidating a particular codeinstance, only invalidate
            // this backedge it actually has an edge for our codeinstance.
            jl_svec_t *edges = jl_atomic_load_relaxed(&replaced->edges);
            int result_edge = 0;
            for (size_t j = 0; j < jl_svec_len(edges); ++j) {
                jl_value_t *edge = jl_svecref(edges, j);
                if (edge == (jl_value_t*)replaced_mi || edge == (jl_value_t*)replaced_ci) {
                    // the callee was invalidated through its own edges: a caller whose only edge to
                    // it is a result edge calls it through a slot, and is marked (compiler §3.3)
                    result_edge = edge == (jl_value_t*)replaced_ci && invokesig == NULL &&
                        jl_code_slot_on() && jl_ci_has_result_edge(replaced, replaced_mi);
                    goto found;
                }
            }
            ins = set_next_edge(backedges, ins, invokesig, replaced);
            continue;
        found:;
            if (result_edge) {
                // the caller keeps its backedge: revalidation names the callee's new CodeInstance
                ins = set_next_edge(backedges, ins, invokesig, replaced);
                mark_code_instance(replaced, max_world, depth);
                if (!replaced_mi->backedges)
                    break;
                continue;
            }
            ins = clear_next_edge(backedges, ins, invokesig, replaced);
            jl_atomic_fetch_or(&replaced_mi->flags, MI_FLAG_BACKEDGES_DIRTY);
            /* fallthrough */
        }
        invalidate_code_instance(replaced, max_world, depth);
        if (replaced_ci && !replaced_mi->backedges) {
            // Fast-path early out. If `invalidate_code_instance` invalidated
            // the entire mi via a recursive edge, there's no point to keep
            // iterating - they'll already have been invalidated.
            break;
        }
    }
    if (replaced_ci)
        jl_mi_done_backedges(replaced_mi, recursion_flags);
    JL_GC_POP();
}

static int jl_type_intersection2(jl_value_t *t1, jl_value_t *t2, jl_value_t **isect JL_REQUIRE_ROOTED_SLOT, jl_value_t **isect2 JL_REQUIRE_ROOTED_SLOT)
{
    *isect2 = NULL;
    int is_subty = 0;
    *isect = jl_type_intersection_env_s(t1, t2, NULL, &is_subty);
    if (*isect == jl_bottom_type)
        return 0;
    if (is_subty)
        return 1;
    // TODO: sometimes type intersection returns types with free variables
    if (jl_has_free_typevars(t1) || jl_has_free_typevars(t2))
        return 1;
    // determine if type-intersection can be convinced to give a better, non-bad answer
    // if the intersection was imprecise, see if we can do better by switching the types
    *isect2 = jl_type_intersection(t2, t1);
    if (*isect2 == jl_bottom_type) {
        *isect = jl_bottom_type;
        *isect2 = NULL;
        return 0;
    }
    if (jl_types_egal(*isect2, *isect)) {
        *isect2 = NULL;
    }
    return 1;
}


// check if `type` is replacing `m` with an ambiguity here, given other methods in `d` that already match it
static int is_replacing(char ambig, jl_value_t *type, jl_method_t *m, jl_method_t *const *d, size_t n, jl_value_t *isect, jl_value_t *isect2, char *morespec)
{
    size_t k;
    for (k = 0; k < n; k++) {
        jl_method_t *m2 = d[k];
        // see if m2 also fully covered this intersection
        if (m == m2 || !(jl_subtype(isect, m2->sig) || (isect2 && jl_subtype(isect2, m2->sig))))
            continue;
        if (morespec[k])
            // not actually shadowing this--m2 will still be better
            return 0;
        // if type is not more specific than m (thus now dominating it)
        // then there is a new ambiguity here,
        // since m2 was also a previous match over isect,
        // see if m was previously dominant over all m2
        // or if this was already ambiguous before
        if (ambig && !jl_type_morespecific(m->sig, m2->sig)) {
            // m and m2 were previously ambiguous over the full intersection of mi with type, and will still be ambiguous with addition of type
            return 0;
        }
    }
    return 1;
}

static int _invalidate_dispatch_backedges(jl_method_instance_t *mi, jl_value_t *type, jl_method_t *m,
        jl_method_t *const *d, size_t n, int replaced_dispatch, int ambig,
        size_t max_world, char *morespec)
{
    uint8_t backedge_recursion_flags = 0;
    jl_array_t *backedges = jl_mi_get_backedges_mutate(mi, &backedge_recursion_flags);
    if (!backedges)
        return 0;
    size_t ib = 0, insb = 0, nb = jl_array_nrows(backedges);
    jl_value_t *invokeTypes;
    jl_code_instance_t *caller;
    int invalidated_any = 0;
    while (mi->backedges && ib < nb) {
        ib = get_next_edge(backedges, ib, &invokeTypes, &caller);
        if (!caller) {
            insb = ib;
            continue;
        }
        JL_GC_PROMISE_ROOTED(caller); // propagated by get_next_edge from backedges
        int replaced_edge;
        if (invokeTypes) {
            // n.b. normally we must have mi.specTypes <: invokeTypes <: m.sig (though it might not strictly hold), so we only need to check the other subtypes
            if (jl_egal(invokeTypes, jl_get_ci_mi(caller)->def.method->sig))
                replaced_edge = 0; // if invokeTypes == m.sig, then the only way to change this invoke is to replace the method itself
            else
                replaced_edge = jl_subtype(invokeTypes, type) && is_replacing(ambig, type, m, d, n, invokeTypes, NULL, morespec);
        }
        else {
            replaced_edge = replaced_dispatch;
        }
        if (replaced_edge) {
            invalidate_code_instance(caller, max_world, 1);
            insb = clear_next_edge(backedges, insb, invokeTypes, caller);
            jl_atomic_fetch_or(&mi->flags, MI_FLAG_BACKEDGES_DIRTY);
            invalidated_any = 1;
        }
        else {
            insb = set_next_edge(backedges, insb, invokeTypes, caller);
        }
    }
    jl_mi_done_backedges(mi, backedge_recursion_flags);
    return invalidated_any;
}

// invalidate cached methods that overlap this definition
static void invalidate_backedges(jl_method_instance_t *replaced_mi, size_t max_world, const char *why)
{
    // Reset dispatch_status when method instance is replaced
    JL_LOCK(&replaced_mi->def.method->writelock);
    _invalidate_backedges(replaced_mi, NULL, max_world, 1);
    JL_UNLOCK(&replaced_mi->def.method->writelock);
    if (why && _jl_debug_method_invalidation) {
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)replaced_mi);
        jl_value_t *loctag = jl_cstr_to_string(why);
        JL_GC_PUSH1(&loctag);
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, loctag);
        JL_GC_POP();
    }
    jl_atomic_store_relaxed(&replaced_mi->dispatch_status, 0);
}

// add a backedge from callee to caller
JL_DLLEXPORT void jl_method_instance_add_backedge(jl_method_instance_t *callee, jl_value_t *invokesig, jl_code_instance_t *caller)
{
    if (!jl_atomic_load_relaxed(&allow_new_worlds))
        return;
    if (invokesig == jl_nothing)
        invokesig = NULL;      // julia uses `nothing` but C uses NULL (#undef)
    assert(jl_is_method_instance(callee));
    assert(jl_is_code_instance(caller));
    assert(invokesig == NULL || jl_is_type(invokesig));
    JL_LOCK(&callee->def.method->writelock);
    if (jl_atomic_load_relaxed(&allow_new_worlds)) {
        jl_array_t *backedges = jl_mi_get_backedges(callee);
        // TODO: use jl_cache_type_(invokesig) like cache_method does to save memory
        if (!backedges) {
            // lazy-init the backedges array
            backedges = jl_alloc_vec_any(0);
            callee->backedges = backedges;
            jl_gc_wb(callee, backedges);
        }
        push_edge(backedges, invokesig, caller);
    }
    JL_UNLOCK(&callee->def.method->writelock);
}


static int jl_foreach_top_typename_for(void (*f)(jl_typename_t*, int, void*), jl_value_t *argtypes JL_PROPAGATES_ROOT, int all_subtypes, void *env);

struct _typename_add_backedge {
    jl_value_t *typ;
    jl_value_t *caller;
};

static void _typename_add_backedge(jl_typename_t *tn, int explct, void *env0)
{
    struct _typename_add_backedge *env = (struct _typename_add_backedge*)env0;
    JL_GC_PROMISE_ROOTED(env->typ);
    JL_GC_PROMISE_ROOTED(env->caller);
    if (!explct)
        return;
    jl_genericmemory_t *allbackedges = jl_method_table->backedges;
    jl_array_t *backedges = (jl_array_t*)jl_eqtable_get(allbackedges, (jl_value_t*)tn, NULL);
    if (backedges == NULL) {
        backedges = jl_alloc_vec_any(2);
        JL_GC_PUSH1(&backedges);
        jl_array_del_end(backedges, 2);
        jl_genericmemory_t *newtable = jl_eqtable_put(allbackedges, (jl_value_t*)tn, (jl_value_t*)backedges, NULL);
        JL_GC_POP();
        if (newtable != allbackedges) {
            jl_method_table->backedges = newtable;
            jl_gc_wb(jl_method_table, newtable);
        }
    }
    // check if the edge is already present and avoid adding a duplicate
    size_t i, l = jl_array_nrows(backedges);
    // reuse an already cached instance of this type, if possible
    // TODO: use jl_cache_type_(tt) like cache_method does, instead of this linear scan?
    // TODO: use as_global_root and de-dup edges array too
    for (i = 1; i < l; i += 2) {
        if (jl_array_ptr_ref(backedges, i) == env->caller) {
            if (jl_types_equal(jl_array_ptr_ref(backedges, i - 1), env->typ)) {
                env->typ = jl_array_ptr_ref(backedges, i - 1);
                return; // this edge already recorded
            }
        }
    }
    for (i = 1; i < l; i += 2) {
        if (jl_array_ptr_ref(backedges, i) != env->caller) {
            if (jl_types_equal(jl_array_ptr_ref(backedges, i - 1), env->typ)) {
                env->typ = jl_array_ptr_ref(backedges, i - 1);
                break;
            }
        }
    }
    jl_array_ptr_1d_push(backedges, env->typ);
    jl_array_ptr_1d_push(backedges, env->caller);
}

// add a backedge from a non-existent signature to caller
JL_DLLEXPORT void jl_method_table_add_backedge(jl_value_t *typ, jl_code_instance_t *caller)
{
    assert(jl_is_code_instance(caller));
    if (!jl_atomic_load_relaxed(&allow_new_worlds))
        return;
    // try to pick the best cache(s) for this typ edge
    jl_methtable_t *mt = jl_method_table;
    jl_methcache_t *mc = mt->cache;
    JL_LOCK(&mc->writelock);
    if (jl_atomic_load_relaxed(&allow_new_worlds)) {
        struct _typename_add_backedge env = {typ, (jl_value_t*)caller};
        jl_foreach_top_typename_for(_typename_add_backedge, typ, 0, &env);
    }
    JL_UNLOCK(&mc->writelock);
}

struct _typename_invalidate_backedge {
    jl_value_t *type;
    jl_value_t *winsig; // M2a fix round (fix 4): the sig answering the winner
                        // (more-specific) question against env->type — equal to it
                        // for insertions, the method's ORIGINAL sig for a §7 replay
    jl_value_t **isect;
    jl_value_t **isect2;
    jl_method_t *const *d;
    size_t n;
    size_t max_world;
    int invalidated;
    jl_method_t *setmethod; // C-73: a method at an open set being inserted (not a replay), else NULL
};

static int shadow_at(jl_value_t *tt, jl_method_t *m0, jl_method_t *mz, size_t i);

static void _typename_invalidate_backedges(jl_typename_t *tn, int explct, void *env0)
{
    struct _typename_invalidate_backedge *env = (struct _typename_invalidate_backedge*)env0;
    JL_GC_PROMISE_ROOTED(env->type);
    JL_GC_PROMISE_ROOTED(env->isect); // isJuliaType considers jl_value_t** to be a julia object too
    JL_GC_PROMISE_ROOTED(env->isect2); // isJuliaType considers jl_value_t** to be a julia object too
    JL_GC_PROMISE_ROOTED(env->winsig); // set alongside type by every construction below
    jl_array_t *backedges = (jl_array_t*)jl_eqtable_get(jl_method_table->backedges, (jl_value_t*)tn, NULL);
    if (backedges == NULL)
        return;
    jl_value_t **d = jl_array_ptr_data(backedges);
    size_t i, na = jl_array_nrows(backedges);
    size_t ins = 0;
    for (i = 1; i < na; i += 2) {
        jl_value_t *backedgetyp = d[i - 1];
        JL_GC_PROMISE_ROOTED(backedgetyp);
        int missing = 0;
        if (jl_type_intersection2(backedgetyp, (jl_value_t*)env->type, env->isect, env->isect2)) {
            // See if the intersection was actually already fully
            // covered, but that the new method is ambiguous.
            //  -> no previous method: now there is one, need to update the missing edge
            //  -> one+ previously matching method(s):
            //    -> more specific then all of them: need to update the missing edge
            //      -> some may have been ambiguous: now there is a replacement
            //      -> some may have been called: now there is a replacement (also will be detected in the loop later)
            //    -> less specific or ambiguous with any one of them: can ignore the missing edge (not missing)
            //      -> some may have been ambiguous: still are
            //      -> some may have been called: they may be partly replaced (will be detected in the loop later)
            // c.f. `is_replacing`, which is a similar query, but with an existing method match to compare against
            missing = 1;
            for (size_t j = 0; j < env->n; j++) {
                jl_method_t *m = env->d[j];
                JL_GC_PROMISE_ROOTED(m);
                if (jl_subtype(*env->isect, m->sig) || (*env->isect2 && jl_subtype(*env->isect2, m->sig))) {
                    // We now know that there actually was a previous
                    // method for this part of the type intersection.
                    if (!jl_type_morespecific(env->winsig, m->sig)) {
                        missing = 0;
                        break;
                    }
                }
            }
        }
        else if (env->setmethod != NULL && jl_is_datatype(backedgetyp)) {
            // C-73 (compiler §5.6): the edge's concrete argument is not a member of the new method's set,
            // so the exact intersection is empty; but it could join later, and the caller that pruned this
            // arm (no method there) holds no guard for it. Invalidated, it compiles again with the guard.
            for (size_t k = 1; k < jl_nparams(backedgetyp) && !missing; k++)
                missing = shadow_at(backedgetyp, NULL, env->setmethod, k);
        }
        *env->isect = *env->isect2 = NULL;
        if (missing) {
            jl_code_instance_t *backedge = (jl_code_instance_t*)d[i];
            JL_GC_PROMISE_ROOTED(backedge);
            invalidate_code_instance(backedge, env->max_world, 0);
            env->invalidated = 1;
            if (_jl_debug_method_invalidation)
                jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)backedgetyp);
        }
        else {
            d[ins++] = d[i - 1];
            d[ins++] = d[i - 0];
        }
    }
    if (ins == 0)
        jl_eqtable_pop(jl_method_table->backedges, (jl_value_t*)tn, NULL, NULL);
    else if (na != ins)
        jl_array_del_end(backedges, na - ins);
}

static void invalidate_missing_backedges(struct _typename_invalidate_backedge *env)
{
    if (!jl_foreach_top_typename_for(_typename_invalidate_backedges, env->type, 1, env)) {
        jl_genericmemory_t *allbackedges = jl_method_table->backedges;
        for (size_t i = 0, n = allbackedges->length; i < n; i += 2) {
            jl_value_t *tn = jl_genericmemory_ptr_ref(allbackedges, i);
            jl_value_t *backedges = jl_genericmemory_ptr_ref(allbackedges, i+1);
            if (tn && tn != jl_nothing && backedges)
                _typename_invalidate_backedges((jl_typename_t*)tn, 0, env);
        }
    }
}

struct invalidate_mt_env {
    jl_typemap_entry_t *newentry;
    jl_array_t *shadowed;
    size_t max_world;
    struct membership_stats *stats; // §7: non-NULL counts the lowerings (walk volume)
};
static int invalidate_mt_cache(jl_typemap_entry_t *oldentry, void *closure0)
{
    struct invalidate_mt_env *env = (struct invalidate_mt_env*)closure0;
    JL_GC_PROMISE_ROOTED(env->newentry);
    if (jl_atomic_load_relaxed(&oldentry->max_world) == ~(size_t)0) {
        jl_method_instance_t *mi = oldentry->func.linfo;
        int intersects = 0;
        jl_method_instance_t **d = (jl_method_instance_t**)jl_array_ptr_data(env->shadowed);
        size_t i, n = jl_array_nrows(env->shadowed);
        for (i = 0; i < n; i++) {
            if (mi == d[i]) {
                intersects = 1;
                break;
            }
        }
        if (intersects && (jl_value_t*)oldentry->sig != mi->specTypes) {
            // the entry may point to a widened MethodInstance, in which case it is worthwhile to check if the new method
            // actually has any meaningful intersection with the old one
            intersects = !jl_has_empty_intersection((jl_value_t*)oldentry->sig, (jl_value_t*)env->newentry->sig);
        }
        if (intersects && oldentry->guardsigs != jl_emptysvec) {
            // similarly, if it already matches an existing guardsigs, this is already safe to keep
            size_t i, l;
            for (i = 0, l = jl_svec_len(oldentry->guardsigs); i < l; i++) {
                // see corresponding code in jl_typemap_entry_assoc_exact
                if (jl_subtype((jl_value_t*)env->newentry->sig, jl_svecref(oldentry->guardsigs, i))) {
                    intersects = 0;
                    break;
                }
            }
        }
        if (intersects) {
            if (_jl_debug_method_invalidation) {
                jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)mi);
                jl_value_t *loctag = jl_cstr_to_string("invalidate_mt_cache");
                JL_GC_PUSH1(&loctag);
                jl_array_ptr_1d_push(_jl_debug_method_invalidation, loctag);
                JL_GC_POP();
            }
            jl_atomic_store_relaxed(&oldentry->max_world, env->max_world);
            if (env->stats)
                env->stats->cache_lowerings++;
        }
    }
    return 1;
}

struct disable_mt_env {
    jl_method_t *replaced;
    size_t max_world;
};
static int disable_mt_cache(jl_typemap_entry_t *oldentry, void *closure0)
{
    struct disable_mt_env *env = (struct disable_mt_env*)closure0;
    if (jl_atomic_load_relaxed(&oldentry->max_world) < ~(size_t)0)
        return 1;
    jl_method_t *m = oldentry->func.linfo->def.method;
    if (m == env->replaced)
        jl_atomic_store_relaxed(&oldentry->max_world, env->max_world);
    return 1;
}

static int typemap_search(jl_typemap_entry_t *entry, void *closure)
{
    if ((void*)(entry->func.method) == *(jl_method_t**)closure) {
        *(jl_typemap_entry_t**)closure = entry;
        return 0;
    }
    return 1;
}

static jl_typemap_entry_t *do_typemap_search(jl_methtable_t *mt JL_PROPAGATES_ROOT, jl_method_t *method) {
    jl_value_t *closure = (jl_value_t*)(method);
    if (jl_typemap_visitor(jl_atomic_load_relaxed(&mt->defs), typemap_search, &closure))
        jl_error("method not in method table");
    return (jl_typemap_entry_t *)closure;
}

static void _method_table_invalidate(jl_methcache_t *mc, void *env0)
{
    // drop this method from mc->cache
    jl_typemap_visitor(jl_atomic_load_relaxed(&mc->cache), disable_mt_cache, env0);
    jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
    size_t i, l = leafcache->length;
    for (i = 1; i < l; i += 2) {
        jl_typemap_entry_t *oldentry = (jl_typemap_entry_t*)jl_genericmemory_ptr_ref(leafcache, i);
        if (oldentry) {
            while ((jl_value_t*)oldentry != jl_nothing) {
                disable_mt_cache(oldentry, env0);
                oldentry = jl_atomic_load_relaxed(&oldentry->next);
            }
        }
    }
}

// Invalidate everything compiled against `replaced`. `deleted` is set when the
// method is being removed outright rather than replaced by one with the same
// signature.
//
// Review W2-3 split: the core (backedges + caches) is separated from the
// method-table entry protocol. jl_method_table_activate/delete keep the full
// wrapper — including the precompile guard and the counter-equality debug
// assert, which pins the INSERTION protocol (review B re-check: activate asserts
// the method's world == counter + 1, walks and invalidates the prior-world table
// while the counter still sits at max_world, and publishes the counter only
// afterwards — so inside the wrapper counter == max_world holds; the round-1
// wording "publishes its world first" had the order backwards). The §7
// conservative path calls
// the core directly: the single-bump declare protocol has already published
// the counter to entry_min_world while max_world = entry_min_world - 1, so the
// equality assert would abort every conservative declare in debug builds, and
// the precompile jl_error would longjmp out of the held world_counter_lock.
static void jl_method_invalidate_core(jl_method_t *replaced, size_t max_world, int deleted)
{
    assert(!replaced->is_for_opaque_closure);
    // Invalidate the backedges
    int invalidated = 0;
    jl_value_t *specializations = jl_atomic_load_relaxed(&replaced->specializations);
    JL_GC_PUSH1(&specializations);
    if (!jl_is_svec(specializations))
        specializations = (jl_value_t*)jl_svec1(specializations);
    size_t i, l = jl_svec_len(specializations);
    for (i = 0; i < l; i++) {
        jl_method_instance_t *mi = (jl_method_instance_t*)jl_svecref(specializations, i);
        if ((jl_value_t*)mi != jl_nothing) {
            invalidated = 1;
            invalidate_backedges(mi, max_world, "jl_method_table_disable");
        }
    }

    jl_methtable_t *mt = jl_method_get_table(replaced);
    struct disable_mt_env mt_cache_env;
    mt_cache_env.max_world = max_world;
    mt_cache_env.replaced = replaced;
    _method_table_invalidate(mt->cache, &mt_cache_env);
    if (deleted) {  // xref #63224; ambiguous callees may not leave backedges, so scan the signature edges too
        jl_value_t *isect = NULL, *isect2 = NULL;
        JL_GC_PUSH2(&isect, &isect2);
        jl_methcache_t *mc = jl_method_table->cache;
        JL_LOCK(&mc->writelock);
        struct _typename_invalidate_backedge typename_env = {(jl_value_t*)replaced->sig, (jl_value_t*)replaced->sig, &isect, &isect2, NULL, 0, max_world, 0, NULL};
        invalidate_missing_backedges(&typename_env);
        JL_UNLOCK(&mc->writelock);
        JL_GC_POP();
        invalidated |= typename_env.invalidated;
    }
    JL_GC_POP();
    if (invalidated && _jl_debug_method_invalidation) {
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)replaced);
        jl_value_t *loctag = jl_cstr_to_string("jl_method_table_disable");
        JL_GC_PUSH1(&loctag);
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, loctag);
        JL_GC_POP();
    }
}

static void jl_method_table_invalidate(jl_method_t *replaced, size_t max_world, int deleted)
{
    if (jl_options.incremental && jl_generating_output())
        jl_error("Method deletion is not possible during Module precompile.");
    assert(jl_atomic_load_relaxed(&jl_world_counter) == max_world);
    jl_method_invalidate_core(replaced, max_world, deleted);
}

static int erase_method_backedges(jl_typemap_entry_t *def, void *closure)
{
    jl_method_t *method = def->func.method;
    JL_LOCK(&method->writelock);
    jl_value_t *specializations = jl_atomic_load_relaxed(&method->specializations);
    if (jl_is_svec(specializations)) {
        size_t i, l = jl_svec_len(specializations);
        for (i = 0; i < l; i++) {
            jl_method_instance_t *mi = (jl_method_instance_t*)jl_svecref(specializations, i);
            if ((jl_value_t*)mi != jl_nothing) {
                mi->backedges = 0;
            }
        }
    }
    else {
        jl_method_instance_t *mi = (jl_method_instance_t*)specializations;
        mi->backedges = 0;
    }
    JL_UNLOCK(&method->writelock);
    return 1;
}

static int erase_all_backedges(jl_methtable_t *mt, void *env)
{
    return jl_typemap_visitor(jl_atomic_load_relaxed(&mt->defs), erase_method_backedges, env);
}

JL_DLLEXPORT void jl_disable_new_worlds(void)
{
    if (jl_generating_output())
        jl_error("Disabling Method changes is not possible when generating output.");
    JL_LOCK(&world_counter_lock);
    jl_atomic_store_relaxed(&allow_new_worlds, 0);
    JL_UNLOCK(&world_counter_lock);
    jl_array_t *mod_array = jl_get_loaded_modules();
    JL_GC_PUSH1(&mod_array);
    jl_foreach_reachable_mtable(erase_all_backedges, mod_array, (void*)NULL);

    JL_LOCK(&jl_method_table->cache->writelock);
    jl_method_table->backedges = (jl_genericmemory_t*)jl_an_empty_memory_any;
    JL_UNLOCK(&jl_method_table->cache->writelock);
    JL_GC_POP();
}

// A method of the callable type `f` deleted can end its membership in Callable (compiler §7.4): the
// dispatch cache entries that selected a method at Callable for `f` are lowered, so the next call
// looks up again. Under the world counter's lock.
struct lower_callable_env {
    jl_value_t *f;
    size_t max_world;
    jl_typename_t *cname;
};
static int cache_entry_through_callable(jl_typemap_entry_t *e, struct lower_callable_env *env)
{
    jl_value_t *v = e->func.value;
    jl_method_t *m = jl_is_method_instance(v) ? ((jl_method_instance_t*)v)->def.method :
                     jl_is_method(v) ? (jl_method_t*)v : NULL;
    if (m == NULL || !jl_is_method(m) || !jl_has_open_set_in(m->sig))
        return 0;
    jl_value_t *sv = jl_unwrap_unionall((jl_value_t*)e->sig), *d = jl_unwrap_unionall(m->sig);
    if (!jl_is_datatype(sv) || !jl_is_datatype(d))
        return 0;
    int hit = 0;
    for (size_t i = 1; i < jl_nparams(sv) && !hit; i++) {
        if (jl_tparam(sv, i) != env->f)
            continue;
        jl_value_t *di = sig_arg(d, i, m->isva);
        arraylist_t names;
        arraylist_new(&names, 0);
        set_names_in(di, &names, 0);
        for (size_t k = 0; k < names.len; k++)
            hit |= names.items[k] == env->cname;
        arraylist_free(&names);
    }
    return hit;
}
static int lower_callable_entry(jl_typemap_entry_t *e, void *closure0)
{
    struct lower_callable_env *env = (struct lower_callable_env*)closure0;
    if (jl_atomic_load_relaxed(&e->max_world) > env->max_world && cache_entry_through_callable(e, env))
        jl_atomic_store_relaxed(&e->max_world, env->max_world);
    return 1;
}
static void lower_callable_dispatch(jl_method_t *method, size_t max_world)
{
    if (jl_callable_type == NULL)
        return;
    jl_value_t *msig = jl_unwrap_unionall(method->sig);
    if (!jl_is_datatype(msig) || jl_nparams(msig) == 0)
        return;
    jl_value_t *f = jl_tparam0(msig);
    if (!jl_is_concrete_type(f))
        return;
    struct lower_callable_env env = {f, max_world,
        ((jl_datatype_t*)jl_unwrap_unionall((jl_value_t*)jl_callable_type))->name};
    jl_methcache_t *mc = jl_method_table->cache;
    JL_LOCK(&mc->writelock);
    jl_typemap_visitor(jl_atomic_load_relaxed(&mc->cache), lower_callable_entry, (void*)&env);
    jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
    for (size_t i = 1; i < leafcache->length; i += 2) {
        jl_value_t *entry = jl_genericmemory_ptr_ref(leafcache, i);
        while (entry != NULL && entry != jl_nothing) {
            lower_callable_entry((jl_typemap_entry_t*)entry, (void*)&env);
            entry = (jl_value_t*)jl_atomic_load_relaxed(&((jl_typemap_entry_t*)entry)->next);
        }
    }
    JL_UNLOCK(&mc->writelock);
}

JL_DLLEXPORT void jl_method_table_disable(jl_method_t *method)
{
    jl_call_deletion_hook(method); // strict mode may refuse it (compiler §10.2)
    jl_methtable_t *mt = jl_method_get_table(method);
    jl_typemap_entry_t *methodentry = do_typemap_search(mt, method);
    JL_LOCK(&world_counter_lock);
    if (!jl_atomic_load_relaxed(&allow_new_worlds))
        jl_error("Method changes have been disabled via a call to disable_new_worlds.");
    int enabled = jl_atomic_load_relaxed(&methodentry->max_world) == ~(size_t)0;
    if (enabled) {
        // Narrow the world age on the method to make it uncallable
        size_t world = jl_atomic_load_relaxed(&jl_world_counter);
        assert(method == methodentry->func.method);
        jl_atomic_store_relaxed(&method->dispatch_status, 0);
        assert(jl_atomic_load_relaxed(&methodentry->max_world) == ~(size_t)0);
        jl_atomic_store_relaxed(&methodentry->max_world, world);
        jl_method_table_invalidate(method, world, 1);
        lower_callable_dispatch(method, world);
        jl_atomic_store_release(&jl_world_counter, world + 1);
    }
    JL_UNLOCK(&world_counter_lock);
    if (!enabled)
        jl_errorf("Method of %s already disabled", jl_symbol_name(method->name));
}

jl_typemap_entry_t *jl_method_table_add(jl_methtable_t *mt, jl_method_t *method, jl_tupletype_t *simpletype)
{
    JL_TIMING(ADD_METHOD, ADD_METHOD);
    assert(jl_is_method(method));
    assert(jl_is_mtable(mt));
    jl_timing_show_method(method, JL_TIMING_DEFAULT_BLOCK);
    jl_typemap_entry_t *newentry = NULL;
    JL_GC_PUSH1(&newentry);
    // add our new entry
    assert(jl_atomic_load_relaxed(&method->primary_world) == ~(size_t)0); // min-world
    assert((jl_atomic_load_relaxed(&method->dispatch_status) & METHOD_SIG_LATEST_WHICH) == 0);
    assert((jl_atomic_load_relaxed(&method->dispatch_status) & METHOD_SIG_LATEST_ONLY) == 0);
    JL_LOCK(&mt->cache->writelock);
    newentry = jl_typemap_alloc((jl_tupletype_t*)method->sig, simpletype, jl_emptysvec, (jl_value_t*)method, ~(size_t)0, 1);
    jl_typemap_insert(&mt->defs, (jl_value_t*)mt, newentry, 0);

    // index the methods whose signatures name an open set (review M10)
    jl_index_open_set_methods(method);

    if (mt == jl_method_table)
        update_max_args(method->sig);
    JL_UNLOCK(&mt->cache->writelock);
    JL_GC_POP();
    return newentry;
}

static int has_key(jl_genericmemory_t *keys, jl_value_t *key)
{
    for (size_t l = keys->length, i = 0; i < l; i++) {
        jl_value_t *k = jl_genericmemory_ptr_ref(keys, i);
        if (k == NULL)
            return 0;
        if (jl_genericmemory_ptr_ref(keys, i) == key)
            return 1;
    }
    return 0;
}

// a signature's open-set bit (its tuple type's, which covers its variables' bounds); inline
STATIC_INLINE int sig_has_open_set(jl_value_t *sig) JL_NOTSAFEPOINT
{
    while (jl_is_unionall(sig))
        sig = ((jl_unionall_t*)sig)->body;
    return jl_is_datatype(sig) && ((jl_datatype_t*)sig)->has_open_set;
}

// Check if m2 is in m1's interferences set, which means !morespecific(m1, m2).
// The set is recorded when a method is inserted. A pair with a method at an open set is asked
// in the query's world instead (compiler §5.2): a later membership can make two methods meet that
// were disjoint at insertion (a method at a set and one at a type that later joins it), or order
// two sets, and the record knows neither.
static int method_in_interferences(jl_method_t *m2, jl_method_t *m1)
{
    if (sig_has_open_set((jl_value_t*)m1->sig) || sig_has_open_set((jl_value_t*)m2->sig))
        return !jl_method_morespecific(m1, m2);
    return has_key(jl_atomic_load_relaxed(&m1->interferences), (jl_value_t*)m2);
}

// Find the index of a method in the method match array
static int find_method_in_matches(jl_array_t *t, jl_method_t *method)
{
    size_t len = jl_array_nrows(t);
    for (size_t i = 0; i < len; i++) {
        jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(t, i);
        if (matc->method == method)
            return i;
    }
    return -1;
}

// Recursively check if any method in interferences covers the given type signature
static int check_interferences_covers(jl_method_t *m, jl_value_t *ti, jl_array_t *t, arraylist_t *visited, arraylist_t *seen)
{
    arraylist_t workqueue;
    arraylist_new(&workqueue, 0);
    arraylist_push(&workqueue, m);
    arraylist_push(seen, (void*)m);
    int result = 0;
    while (workqueue.len > 0) {
        jl_method_t *current_m = (jl_method_t*)arraylist_pop(&workqueue);
        JL_GC_PROMISE_ROOTED(current_m);
        jl_genericmemory_t *interferences = jl_atomic_load_relaxed(&current_m->interferences);
        for (size_t i = 0; i < interferences->length; i++) {
            jl_method_t *m2 = (jl_method_t*)jl_genericmemory_ptr_ref(interferences, i);
            if (m2 == NULL)
                continue;
            // Check if we already visited this method
            int in_seen = 0;
            for (size_t i = 0; i < seen->len; i++) {
                if (seen->items[i] == (void*)m2) {
                    in_seen = 1;
                    break;
                }
            }
            if (in_seen)
                continue;
            arraylist_push(seen, (void*)m2);
            int idx = find_method_in_matches(t, m2);
            if (idx < 0)
                continue;
            if (method_in_interferences(current_m, m2))
                continue; // ambiguous
            assert(visited->items[idx] != (void*)0);
            if (visited->items[idx] != (void*)1)
                continue; // part of the same SCC cycle (handled by ambiguity later)
            if (jl_subtype(ti, m2->sig)) {
                result = 1;
                goto cleanup;
            }
            arraylist_push(&workqueue, m2);
        }
    }
cleanup:
    seen->len = 0;
    arraylist_free(&workqueue);
    return result;
}

static int check_fully_ambiguous(jl_method_t *m, jl_value_t *ti, jl_array_t *t, int include_ambiguous, int *has_ambiguity)
{
    jl_genericmemory_t *interferences = jl_atomic_load_relaxed(&m->interferences);
    for (size_t i = 0; i < interferences->length; i++) {
        jl_method_t *m2 = (jl_method_t*)jl_genericmemory_ptr_ref(interferences, i);
        if (m2 == NULL)
            continue;
        int idx = find_method_in_matches(t, m2);
        if (idx < 0)
            continue;
        if (!method_in_interferences(m, m2))
            continue;
        *has_ambiguity = 1;
        if (!include_ambiguous && jl_subtype(ti, m2->sig))
            return 1;
    }
    return 0;
}

// Recursively check if target_method is in the interferences of (morespecific than) start_method, but not the reverse
static int method_morespecific_via_interferences(jl_method_t *target_method, jl_method_t *start_method)
{
    if (target_method == start_method)
        return 0;
    // Check direct interferences first
    if (method_in_interferences(start_method, target_method))
        return 0;
    if (method_in_interferences(target_method, start_method))
        return 1;
    arraylist_t seen;
    arraylist_t workqueue;
    arraylist_new(&seen, 0);
    arraylist_push(&seen, (void*)start_method);
    arraylist_new(&workqueue, 0);
    arraylist_push(&workqueue, start_method);
    int result = 0;
    while (workqueue.len > 0) {
        jl_method_t *current = (jl_method_t*)arraylist_pop(&workqueue);
        JL_GC_PROMISE_ROOTED(current);
        jl_genericmemory_t *interferences = jl_atomic_load_relaxed(&current->interferences);
        for (size_t i = 0; i < interferences->length; i++) {
            jl_method_t *interference_method = (jl_method_t*)jl_genericmemory_ptr_ref(interferences, i);
            if (interference_method == NULL)
                continue;
            // Check if we're already visiting this interference method (cycle prevention)
            int already_seen = 0;
            for (size_t j = 0; j < seen.len; j++) {
                if (seen.items[j] == (void*)interference_method) {
                    already_seen = 1;
                    break;
                }
            }
            if (already_seen)
                continue;
            arraylist_push(&seen, interference_method);
            if (method_in_interferences(current, interference_method))
                continue; // only follow edges to morespecific methods in search of morespecific target (skip ambiguities)
            // Check direct interferences for this interference method
            if (method_in_interferences(interference_method, target_method))
                continue; // return 0 for this path
            if (method_in_interferences(target_method, interference_method)) {
                result = 1;
                goto cleanup;
            }
            arraylist_push(&workqueue, interference_method);
        }
    }
cleanup:
    arraylist_free(&workqueue);
    arraylist_free(&seen);
    //assert(result == jl_method_morespecific(target_method, start_method) || jl_has_empty_intersection(target_method->sig, start_method->sig) || jl_has_empty_intersection(start_method->sig, target_method->sig));
    return result;
}


// The body of jl_method_table_activate's "not a replacement" branch, shared with
// the membership walk's replay (§7): re-resolve the methods in d[] that intersect
// `type` against the incoming `method` — update the specificity bookkeeping
// (interferences, the LATEST_ONLY bits) and invalidate the specializations and the
// callers whose selected method changed. Returns whether anything was invalidated.
// *dispatch_bits and **interferences are the caller's in-place state; isect, isect2
// and loctag are its rooted scratch slots.
// replay is set by jl_membership_dispatch_walk: nothing is being inserted, so the
// replayed method's own typemap entry is live and shows up in d[] (skip it), and the
// dispatch_bits publish (which activate does at the end of an insertion) does not
// run. stats, when non-NULL, counts the walk's volume for §5.M2's log.
// Whether a membership's dispatch walk invalidates compiled code (compiler §5.5, §5.6): it lowers the
// dispatch caches either way. Code compiled with guards needs no invalidation, so a membership
// invalidates nothing; the stopgap C-18 invalidated the callers it could change.
static int membership_walk_invalidates = 0;
JL_DLLEXPORT int jl_set_membership_walk_invalidates(int on)
{
    int last = membership_walk_invalidates;
    membership_walk_invalidates = on;
    return last;
}

static int invalidate_intersecting(jl_value_t *type, jl_method_t *method,
                                   jl_method_t *const *d, size_t n, char *morespec,
                                   size_t max_world, jl_array_t *oldmi,
                                   int replay, struct membership_stats *stats,
                                   int *dispatch_bits, jl_genericmemory_t **interferences,
                                   jl_value_t **isect, jl_value_t **isect2, jl_value_t **loctag)
{
    size_t j;
    int invalidated = 0;
    for (j = 0; j < n; j++) {
        jl_method_t *m = d[j];
        if (replay && m == method)
            continue; // the replay's own entry is live in the table, unlike a real insertion
        if (stats)
            stats->visits++;
        // Compute ambig state: is there an ambiguity between new method and old m?
        // M2a fix round (fix 4): the new method's ORIGINAL sig answers this
        // winner question — for a real insertion entry->sig == method->sig, so
        // this is bit-identical there; for the §7 replay `type` is the
        // substituted sig, which only drives the intersections below.
        char ambig = !morespec[j] && !jl_type_morespecific((jl_value_t*)method->sig, m->sig);
        // Compute updates to the dispatch state bits
        int m_dispatch = jl_atomic_load_relaxed(&m->dispatch_status);
        if (morespec[j] || ambig) {
            // !morespecific(new, old)
            *dispatch_bits &= ~METHOD_SIG_LATEST_ONLY;
            // Add the old method to this interference set
            ssize_t idx;
            if (!has_key(*interferences, (jl_value_t*)m))
                *interferences = jl_idset_put_key(*interferences, (jl_value_t*)m, &idx);
        }
        if (!morespec[j]) {
            // !morespecific(old, new)
            if (m_dispatch & METHOD_SIG_LATEST_ONLY) {
                if (stats)
                    stats->latest_only_clears++;
            }
            m_dispatch &= ~METHOD_SIG_LATEST_ONLY;
            // Add the new method to its interference set
            jl_genericmemory_t *m_interferences = jl_atomic_load_relaxed(&m->interferences);
            ssize_t idx;
            m_interferences = jl_idset_put_key(m_interferences, (jl_value_t*)method, &idx);
            jl_atomic_store_release(&m->interferences, m_interferences);
            jl_gc_wb(m, m_interferences);
        }
        // Add methods that intersect but are not more specific to interference list.
        // A replay writes these too: probe 3's stale LATEST_ONLY on a chain-matched
        // method is exactly what the membership walk has to clear.
        jl_atomic_store_relaxed(&m->dispatch_status, m_dispatch);
        if (morespec[j]) {
            if (stats)
                stats->morespec_skips++;
            continue;
        }

        // Now examine if this caused any invalidations.
        *loctag = jl_atomic_load_relaxed(&m->specializations); // use loctag for a gcroot
        _Atomic(jl_method_instance_t*) *data;
        size_t l;
        if (jl_is_svec(*loctag)) {
            data = (_Atomic(jl_method_instance_t*)*)jl_svec_data(*loctag);
            l = jl_svec_len(*loctag);
        }
        else {
            data = (_Atomic(jl_method_instance_t*)*)loctag;
            l = 1;
        }
        for (size_t i = 0; i < l; i++) {
            jl_method_instance_t *mi = jl_atomic_load_relaxed(&data[i]);
            if ((jl_value_t*)mi == jl_nothing)
                continue;
            if (jl_type_intersection2(type, mi->specTypes, isect, isect2)) {
                // Replacing a method--see if this really was the selected method previously
                // over the intersection (not ambiguous) and the new method will be selected now (morespec).
                // TODO: this only checks pair-wise for ambiguities, but the ambiguities could arise from the interaction of multiple methods
                // and thus might miss a case where we introduce an ambiguity between`.u two existing methods
                // We could instead work to sort this into 3 groups `morespecific .. ambiguous .. lesspecific`, with `type` in ambiguous,
                // such that everything in `morespecific` dominates everything in `ambiguous`, and everything in `ambiguous` dominates everything in `lessspecific`
                // And then compute where each isect falls, and whether it changed group--necessitating invalidation--or not.
                int replaced_dispatch = is_replacing(ambig, type, m, d, n, *isect, *isect2, morespec);
                // found that this specialization dispatch got replaced by m
                // call invalidate_backedges(mi, max_world, "jl_method_table_insert");
                // but ignore invoke-type edges
                int invalidatedmi = (!replay || membership_walk_invalidates) ?
                    _invalidate_dispatch_backedges(mi, type, m, d, n, replaced_dispatch, ambig, max_world, morespec) : 0;
                if (replaced_dispatch) {
                    if (stats)
                        stats->spec_invalidations++;
                    jl_atomic_store_relaxed(&mi->dispatch_status, 0);
                    jl_array_ptr_1d_push(oldmi, (jl_value_t*)mi);
                    if (replay && membership_walk_invalidates) {
                        // M2a fix round 2 (gate m2_dispatch_worlds): the answer at
                        // specTypes flips at the entry's world. Lowering the mt-cache
                        // entries (the walk's tail) bounds reachability but leaves
                        // the CI chain's own range unbounded, and the world gates
                        // pin it: the loser's CIs hold the pre-membership answer,
                        // valid exactly through max_world = entry_min_world - 1.
                        // A vanilla insertion (replay = 0) does NOT do this — there
                        // the old CI stays valid for old worlds by the leaf-entry
                        // bound alone; the §7 replay is the conservative walker and
                        // bounds the range here. invalidate_code_instance recurses
                        // into callers' backedges and is idempotent on already
                        // lowered links (second branch asserts monotonicity).
                        jl_code_instance_t *ci = jl_atomic_load_relaxed(&mi->cache);
                        while (ci) {
                            invalidate_code_instance(ci, max_world, 1);
                            ci = jl_atomic_load_relaxed(&ci->next);
                        }
                    }
                }
                if (_jl_debug_method_invalidation && invalidatedmi) {
                    jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)mi);
                    *loctag = jl_cstr_to_string(replay ? "jl_membership_dispatch_walk" : "jl_method_table_insert");
                    jl_array_ptr_1d_push(_jl_debug_method_invalidation, *loctag);
                }
                invalidated |= invalidatedmi;
            }
            // TODO: do we have any interesting cases left where isect3 is useful
            //jl_value_t *isect3 = NULL;
            //jl_value_t *isect4 = NULL;
            //jl_value_t *isect5 = NULL;
            //JL_GC_PUSH3(&isec3, &isect4, &isect5);
            //isect3 = jl_type_intersection(m->sig, (jl_value_t*)mi->specTypes);
            //jl_type_intersection2(type, isect3, &isect4, &isect5);
            //if (!jl_types_equal(isect, isect4) && (!isect2 || !jl_types_equal(isect2, isect4)) &&
            //    (!isect5 || (!jl_types_equal(isect, isect5) && (!isect2 || !jl_types_equal(isect2, isect5))))) {
            //    jl_(type);
            //    jl_(mi->specTypes);
            //    jl_(m->sig);
            //}
            //JL_GC_POP();
            *isect = NULL;
            *isect2 = NULL;
        }
    }
    return invalidated;
}


// A method at a set defined after code that calls its function at a type that could join the set:
// that code took no guard for it (compiler §5.6), so the callers of each instance of the function the
// method shadows are invalidated, and compile again with the guard.
static void invalidate_unguarded_callers(jl_method_t *mz, size_t world, size_t max_world)
{
    jl_value_t *zsig = jl_unwrap_unionall(mz->sig);
    if (!jl_is_datatype(zsig) || jl_nparams(zsig) == 0)
        return;
    jl_value_t *tt = NULL, *ms = NULL, *specs = NULL;
    JL_GC_PUSH3(&tt, &ms, &specs);
    tt = (jl_value_t*)jl_svec2(jl_tparam0(zsig), jl_wrap_vararg((jl_value_t*)jl_any_type, NULL, 1, 0));
    tt = (jl_value_t*)jl_apply_tuple_type((jl_svec_t*)tt, 1);
    size_t min_valid = 1, max_valid = ~(size_t)0;
    ms = jl_matching_methods((jl_tupletype_t*)tt, jl_nothing, -1, 1, max_world, &min_valid, &max_valid, NULL); // the methods before this one
    if (jl_is_array(ms)) {
        for (size_t k = 0; k < jl_array_nrows((jl_array_t*)ms); k++) {
            jl_method_t *m2 = ((jl_method_match_t*)jl_array_ptr_ref((jl_array_t*)ms, k))->method;
            if (m2 == mz)
                continue;
            specs = jl_atomic_load_relaxed(&m2->specializations);
            size_t l = jl_is_svec(specs) ? jl_svec_len(specs) : 1;
            for (size_t i = 0; i < l; i++) {
                jl_value_t *v = jl_is_svec(specs) ? jl_svecref(specs, i) : specs;
                if (v == NULL || v == jl_nothing || !jl_is_method_instance(v))
                    continue;
                jl_method_instance_t *mi = (jl_method_instance_t*)v;
                jl_value_t *st = mi->specTypes;
                if (!jl_is_datatype(st))
                    continue;
                for (size_t a = 1; a < jl_nparams(st); a++) {
                    if (shadow_at(st, m2, mz, a)) {
                        invalidate_backedges(mi, max_world, "jl_method_table_insert (a method at a set)");
                        break;
                    }
                }
            }
        }
    }
    JL_GC_POP();
}

void jl_method_table_activate(jl_typemap_entry_t *newentry)
{
    JL_TIMING(ADD_METHOD, ADD_METHOD);
    jl_method_t *method = newentry->func.method;
    jl_methtable_t *mt = jl_method_get_table(method);
    assert(jl_is_mtable(mt));
    assert(jl_is_method(method));
    // a method at a set joins its function's list, for the shadow query (compiler §5.6); a method of
    // an image is activated here without being added
    if (jl_has_open_set_in((jl_value_t*)method->sig)) {
        JL_LOCK(&open_set_methods_lock);
        index_func_set_method(method);
        JL_UNLOCK(&open_set_methods_lock);
    }
    jl_timing_show_method(method, JL_TIMING_DEFAULT_BLOCK);
    jl_value_t *type = (jl_value_t*)newentry->sig;
    jl_value_t *oldvalue = NULL;
    jl_array_t *oldmi = NULL;
    size_t world = jl_atomic_load_relaxed(&method->primary_world);
    assert(world == jl_atomic_load_relaxed(&jl_world_counter) + 1); // min-world
    assert((jl_atomic_load_relaxed(&method->dispatch_status) & METHOD_SIG_LATEST_WHICH) == 0);
    assert((jl_atomic_load_relaxed(&method->dispatch_status) & METHOD_SIG_LATEST_ONLY) == 0);
    assert(jl_atomic_load_relaxed(&newentry->min_world) == ~(size_t)0);
    assert(jl_atomic_load_relaxed(&newentry->max_world) == 1);
    jl_atomic_store_relaxed(&newentry->min_world, world);
    jl_atomic_store_relaxed(&method->primary_world, world);
    size_t max_world = world - 1;
    jl_value_t *loctag = NULL;  // debug info for invalidation
    jl_value_t *isect = NULL;
    jl_value_t *isect2 = NULL;
    jl_genericmemory_t *interferences = NULL;
    JL_GC_PUSH6(&oldvalue, &oldmi, &loctag, &isect, &isect2, &interferences);
    jl_typemap_entry_t *replaced = NULL;
    // Check what entries this intersects with in the prior world.
    oldvalue = get_intersect_matches(jl_atomic_load_relaxed(&mt->defs), newentry, &replaced, max_world, 0);
    jl_method_t *const *d;
    size_t j, n;
    if (oldvalue == NULL) {
        d = NULL;
        n = 0;
    }
    else {
        assert(jl_is_array(oldvalue));
        d = (jl_method_t**)jl_array_ptr_data(oldvalue);
        n = jl_array_nrows(oldvalue);
        oldmi = jl_alloc_vec_any(0);
    }

    // These get updated from their state stored in the caches files, since content in cache files gets added "all at once".
    int invalidated = 0;
    int dispatch_bits = METHOD_SIG_LATEST_WHICH; // Always set LATEST_WHICH
    // Check precompiled dispatch status bits
    int precompiled_status = jl_atomic_load_relaxed(&method->dispatch_status);
    if (!(precompiled_status & METHOD_SIG_PRECOMPILE_MANY))
        // This will store if this method will be currently the only result that would returned from `ml_matches` given `sig`.
        dispatch_bits |= METHOD_SIG_LATEST_ONLY; // Tentatively set, will be cleared if not applicable
    // Holds the set of all intersecting methods not more specific than this one.
    // Note: this set may be incomplete (may exclude methods whose intersection
    // is covered by another method that is morespecific than both, causing them
    // to have no relevant type intersection for sorting).
    interferences = (jl_genericmemory_t*)jl_atomic_load_relaxed(&method->interferences);
    if (oldvalue) {
        assert(n > 0);
        if (replaced) {
            oldvalue = (jl_value_t*)replaced;
            jl_method_t *m = replaced->func.method;
            invalidated = 1;
            method_overwrite(newentry, m);
            // This is an optimized version of below, given we know the type-intersection is exact
            jl_method_table_invalidate(m, max_world, 0);
            int m_dispatch = jl_atomic_load_relaxed(&m->dispatch_status);
            // Clear METHOD_SIG_LATEST_ONLY and METHOD_SIG_LATEST_WHICH bits
            jl_atomic_store_relaxed(&m->dispatch_status, 0);
            if (!(m_dispatch & METHOD_SIG_LATEST_ONLY))
                dispatch_bits &= ~METHOD_SIG_LATEST_ONLY;
            // Take over the interference list from the replaced method
            jl_genericmemory_t *m_interferences = jl_atomic_load_relaxed(&m->interferences);
            if (interferences->length == 0) {
                interferences = jl_genericmemory_copy(m_interferences);
            }
            else {
                for (size_t i = 0; i < m_interferences->length; i++) {
                    jl_value_t *k = jl_genericmemory_ptr_ref(m_interferences, i);
                    if (k && !has_key(interferences, (jl_value_t*)k)) {
                        ssize_t idx;
                        interferences = jl_idset_put_key(interferences, (jl_value_t*)k, &idx);
                    }
                }
            }
            ssize_t idx;
            m_interferences = jl_idset_put_key(m_interferences, (jl_value_t*)method, &idx);
            jl_atomic_store_release(&m->interferences, m_interferences);
            jl_gc_wb(m, m_interferences);
            for (j = 0; j < n; j++) {
                jl_method_t *m2 = d[j];
                if (m2 && method_in_interferences(m, m2)) {
                    jl_genericmemory_t *m2_interferences = jl_atomic_load_relaxed(&m2->interferences);
                    ssize_t idx;
                    m2_interferences = jl_idset_put_key(m2_interferences, (jl_value_t*)method, &idx);
                    jl_atomic_store_release(&m2->interferences, m2_interferences);
                    jl_gc_wb(m2, m2_interferences);
                }
            }
            loctag = jl_atomic_load_relaxed(&m->specializations); // use loctag for a gcroot
            _Atomic(jl_method_instance_t*) *data;
            size_t l;
            if (jl_is_svec(loctag)) {
                data = (_Atomic(jl_method_instance_t*)*)jl_svec_data(loctag);
                l = jl_svec_len(loctag);
            }
            else {
                data = (_Atomic(jl_method_instance_t*)*) &loctag;
                l = 1;
            }
            for (size_t i = 0; i < l; i++) {
                jl_method_instance_t *mi = jl_atomic_load_relaxed(&data[i]);
                if ((jl_value_t*)mi == jl_nothing)
                    continue;
                jl_array_ptr_1d_push(oldmi, (jl_value_t*)mi);
            }
            d = NULL;
            n = 0;
        }
        else {
            char *morespec = (char*)alloca(n);
            // Compute all morespec values upfront
            for (j = 0; j < n; j++)
                morespec[j] = (char)jl_type_morespecific(d[j]->sig, type);
            invalidated |= invalidate_intersecting(type, method, d, n, morespec, max_world,
                                                   oldmi, 0, NULL, &dispatch_bits,
                                                   &interferences, &isect, &isect2, &loctag);
        }
    }

    jl_methcache_t *mc = jl_method_table->cache;
    JL_LOCK(&mc->writelock);
    // compiler §5.6: a method at an open set reaches the missing edges of arms its set could later
    // shadow; a method that can only raise at a member is no shadow
    jl_method_t *setmethod = (jl_has_open_set_in((jl_value_t*)method->sig) &&
                              !method_raises_only(method)) ? method : NULL;
    struct _typename_invalidate_backedge typename_env = {type, type, &isect, &isect2, d, n, max_world, invalidated, setmethod};
    invalidate_missing_backedges(&typename_env);
    invalidated |= typename_env.invalidated;
    if (jl_has_open_set_in((jl_value_t*)method->sig))
        invalidate_unguarded_callers(method, world, max_world);
    // a method of a callable type can begin its membership in Callable (compiler §7.4)
    {
        jl_value_t *msig = jl_unwrap_unionall((jl_value_t*)method->sig);
        if (jl_is_datatype(msig) && jl_nparams(msig) > 0)
            jl_membership_update_callable_cells(jl_tparam0(msig), world);
    }
    if (oldmi && jl_array_nrows(oldmi)) {
        // drop leafcache and search mc->cache and drop anything that might overlap with the new method
        // this is very cheap, so we don't mind being very conservative at over-approximating this
        struct invalidate_mt_env mt_cache_env;
        mt_cache_env.max_world = max_world;
        mt_cache_env.shadowed = oldmi;
        mt_cache_env.newentry = newentry;
        mt_cache_env.stats = NULL;

        jl_typemap_visitor(jl_atomic_load_relaxed(&mc->cache), invalidate_mt_cache, (void*)&mt_cache_env);
        jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
        size_t i, l = leafcache->length;
        for (i = 1; i < l; i += 2) {
            jl_value_t *entry = jl_genericmemory_ptr_ref(leafcache, i);
            if (entry) {
                while (entry != jl_nothing) {
                    jl_atomic_store_relaxed(&((jl_typemap_entry_t*)entry)->max_world, max_world);
                    entry = (jl_value_t*)jl_atomic_load_relaxed(&((jl_typemap_entry_t*)entry)->next);
                }
            }
        }
        jl_atomic_store_relaxed(&mc->leafcache, (jl_genericmemory_t*)jl_an_empty_memory_any);
    }
    JL_UNLOCK(&mc->writelock);
    if (invalidated && _jl_debug_method_invalidation) {
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)method);
        loctag = jl_cstr_to_string("jl_method_table_insert");
        jl_array_ptr_1d_push(_jl_debug_method_invalidation, loctag);
    }
    jl_atomic_store_relaxed(&newentry->max_world, ~(size_t)0);
    jl_atomic_store_relaxed(&method->dispatch_status, dispatch_bits); // TODO: this should be sequenced fully after the world counter store
    jl_atomic_store_release(&method->interferences, interferences);
    jl_gc_wb(method, interferences);
    JL_GC_POP();
}

// §7 (plan §4.2 item 4, P6 stopgap + permanent cache walk, review H4/§3.2): the
// declaration-time dispatch walk. For a new membership (`member` joins
// `set_instance`, entry visible from entry_min_world), every method whose
// signature names a set the member (transitively) joins is re-resolved as if
// its sig, with the member substituted for the set, were inserted now — the
// replay mode of the else-branch just extracted above. Calling convention
// (single-bump protocol, review W3-F4): called under world_counter_lock with the
// entry published and the counter ALREADY advanced to entry_min_world — the walk
// runs at world == counter, so ml_matches' future guard passes and the
// conservative path uses the counter-free jl_method_invalidate_core (W2-3);
// §0's query-world override is set by the declaring call site for this whole
// region (D1), and the declaring task is synced to entry_min_world (n3), so the
// just-published entry is visible to the replay's intersections. The region is
// written throw-free: a throw under JL_LOCK would leak world_counter_lock (and
// the §0 override) — accepted with the OOM-class risk (D-M2-6).

// sets to walk: set_instance plus its superset closure, over the §1 entry lists
// on each visited set's own TypeName: an entry (M, Y) naming a visited S with
// S <: M witnesses S ⊆ M ⊆ Y, so Y joins the walk too. Lists are tiny and
// jl_declare_membership refuses set-in-set cycles (§1); seen-through-`sets`
// dedup and the caps are belts.
static void membership_superset_sets(jl_value_t *set_instance, size_t entry_min_world, jl_array_t *sets)
{
    jl_array_ptr_1d_push(sets, set_instance);
    size_t i = 0;
    while (i < jl_array_nrows(sets) && jl_array_nrows(sets) < 16) {
        jl_value_t *S = jl_array_ptr_ref(sets, i++);
        jl_value_t *Suw = jl_unwrap_unionall(S);
        if (!jl_is_datatype(Suw))
            continue;
        jl_typename_t *tn = ((jl_datatype_t*)Suw)->name;
        jl_svec_t *list = jl_typename_memberships(tn);
        size_t l = jl_svec_len(list);
        for (size_t j = 0; j < l; j++) {
            jl_membership_t *en = (jl_membership_t*)jl_svecref(list, j);
            if (en->min_world > entry_min_world)
                continue; // not visible in the world this walk answers for
            if (!jl_subtype(S, (jl_value_t*)en->member))
                continue;
            jl_value_t *Y = en->instance;
            size_t k;
            for (k = 0; k < jl_array_nrows(sets); k++)
                if (jl_array_ptr_ref(sets, k) == Y)
                    break;
            if (k == jl_array_nrows(sets))
                jl_array_ptr_1d_push(sets, Y);
        }
    }
}

// does the set's typename occur strictly below a signature position — inside
// parameters, under Vararg, or in a TypeVar bound? Those shapes cannot take a
// positional substitution; the §7 conservative path handles them.
static int set_name_nested(jl_value_t *v, jl_typename_t *tn, int depth)
{
    v = jl_unwrap_unionall(v);
    if (v == NULL || depth > 8)
        return 0;
    if (jl_is_uniontype(v))
        return set_name_nested(((jl_uniontype_t*)v)->a, tn, depth + 1) ||
               set_name_nested(((jl_uniontype_t*)v)->b, tn, depth + 1);
    if (jl_is_vararg(v))
        return set_name_nested(jl_unwrap_vararg(v), tn, depth + 1);
    if (jl_is_typevar(v)) {
        jl_tvar_t *tv = (jl_tvar_t*)v;
        return set_name_nested(tv->lb, tn, depth + 1) ||
               set_name_nested(tv->ub, tn, depth + 1);
    }
    if (jl_is_datatype(v)) {
        jl_datatype_t *dt = (jl_datatype_t*)v;
        if (dt->name == tn)
            return 1;
        for (size_t i = 0; i < jl_nparams(dt); i++)
            if (set_name_nested(jl_tparam(dt, i), tn, depth + 1))
                return 1;
    }
    return 0;
}

// the sig to replay for method m: m's signature with every direct positional
// occurrence of a walked set instance replaced by the member pattern — the
// member's core, re-wrapped over the sig's own UnionAll vars first and the
// member's remaining vars outside them (family declarations share the tvar
// objects, so a shared var binds exactly once — M1's pairing fix). *clean is
// set to 0 when the shape resists substitution (a parametric or free-var
// position of the set's family that the walked set is not a subtype of, or a
// nested/Vararg occurrence); the caller then takes the conservative path.
// can m's parameter `uw` (its free variables m's) be met by a member of the set instance `Y`?
// Exact where both are closed; where either keeps a variable it cannot close, yes (C-72):
// the substituted walk then over-approximates, which lowers more and never misses.
static int membership_param_meets(jl_value_t *Y, jl_value_t *uw, jl_value_t *msig)
{
    if (jl_has_free_typevars(Y))
        return 1; // a pattern's instance
    if (!jl_has_free_typevars(uw))
        return jl_egal(Y, uw) || jl_subtype(Y, uw);
    jl_value_t *w = jl_rewrap_unionall(uw, msig);
    JL_GC_PUSH1(&w);
    int meets = jl_has_free_typevars(w) || !jl_has_empty_intersection(Y, w);
    JL_GC_POP();
    return meets;
}

static jl_value_t *membership_subst_sig(jl_method_t *m, jl_array_t *sets,
                                        jl_value_t *member, int *clean)
{
    jl_value_t *msig = (jl_value_t*)m->sig;
    jl_value_t *mb = jl_unwrap_unionall(msig);
    if (!jl_is_tuple_type(mb)) {
        *clean = 0;
        return NULL;
    }
    jl_value_t *memb = jl_unwrap_unionall(member);
    if (!jl_is_datatype(memb)) {
        *clean = 0;
        return NULL;
    }
    jl_value_t *mvars[16]; size_t mvars_n = 0;
    for (jl_value_t *t = msig; jl_is_unionall(t); t = ((jl_unionall_t*)t)->body) {
        if (mvars_n == 16) {
            *clean = 0;
            return NULL;
        }
        mvars[mvars_n++] = (jl_value_t*)((jl_unionall_t*)t)->var;
    }
    jl_value_t *membvars[16]; size_t membvars_n = 0;
    for (jl_value_t *t = member; jl_is_unionall(t); t = ((jl_unionall_t*)t)->body) {
        if (membvars_n == 16) {
            *clean = 0;
            return NULL;
        }
        membvars[membvars_n++] = (jl_value_t*)((jl_unionall_t*)t)->var;
    }
    jl_datatype_t *mbd = (jl_datatype_t*)mb;
    size_t n = jl_nparams(mbd);
    jl_svec_t *newparams = jl_alloc_svec(n);
    jl_datatype_t *tuple = NULL;
    jl_value_t *res = NULL;
    JL_GC_PUSH3(&newparams, &tuple, &res);
    size_t nreplaced = 0;
    for (size_t i = 0; i < n; i++) {
        jl_value_t *p = jl_tparam(mbd, i);
        jl_value_t *uw = jl_is_vararg(p) ? jl_unwrap_vararg(p) : jl_unwrap_unionall(p);
        if (uw && jl_is_typevar(uw)) // f(c::C) where C <: X{T}: the variable's bound names the set
            uw = jl_unwrap_unionall(((jl_tvar_t*)uw)->ub);
        int replaced = 0;
        int named = 0;
        if (uw && jl_is_datatype(uw)) {
            jl_datatype_t *pd = (jl_datatype_t*)uw;
            for (size_t s = 0; s < jl_array_nrows(sets); s++) {
                jl_value_t *Y = jl_array_ptr_ref(sets, s);
                jl_value_t *Yuw = jl_unwrap_unionall(Y);
                if (((jl_datatype_t*)Yuw)->name != pd->name)
                    continue;
                named = 1;
                if (!membership_param_meets(Y, uw, msig))
                    continue; // another instance of the set: the membership does not reach m here
                if (jl_is_vararg(p)) {
                    // review W2-5c: substituting at a vararg position here would
                    // write the BARE member and strip the Vararg — a malformed
                    // dummy sig. Route to the conservative path instead (making
                    // the driver's "set under Vararg is conservative" claim true).
                    *clean = 0;
                    JL_GC_POP();
                    return NULL;
                }
                // C-72: also where m's parameter has its own variables (X{T}, C <: X{T}): the
                // member in its place, the variables left free elsewhere, which the walk's
                // intersections over-approximate, never miss
                jl_svecset(newparams, i, memb);
                replaced = 1;
                nreplaced++;
                break;
            }
            if (named && !replaced) { // only other instances of the set at this position
                jl_svecset(newparams, i, p);
                continue;
            }
        }
        if (!replaced) {
            for (size_t s = 0; s < jl_array_nrows(sets); s++) {
                jl_value_t *Yuw = jl_unwrap_unionall(jl_array_ptr_ref(sets, s));
                if (set_name_nested(p, ((jl_datatype_t*)Yuw)->name, 1)) {
                    *clean = 0;
                    JL_GC_POP();
                    return NULL;
                }
            }
            jl_svecset(newparams, i, p);
        }
    }
    if (nreplaced == 0) {
        // m names the set at top level only at instances the member is not of (else the
        // nested check above went conservative): its dispatch at the member is unchanged
        *clean = 2;
        JL_GC_POP();
        return NULL;
    }
    // build: tuple, wrapped over m's vars, then the member's vars not already bound there
    tuple = (jl_datatype_t*)jl_apply_tuple_type(newparams, 0);
    res = (jl_value_t*)tuple; // each jl_new_struct below allocates across the previous step
    for (size_t i = mvars_n; i-- > 0;) {
        jl_unionall_t *na = (jl_unionall_t*)jl_new_struct(jl_unionall_type, mvars[i], res);
        res = (jl_value_t*)na;
    }
    for (size_t i = membvars_n; i-- > 0;) {
        int shared = 0;
        for (size_t k = 0; k < mvars_n; k++)
            if (membvars[i] == mvars[k])
                shared = 1;
        if (!shared) {
            jl_unionall_t *na = (jl_unionall_t*)jl_new_struct(jl_unionall_type, membvars[i], res);
            res = (jl_value_t*)na;
        }
    }
    JL_GC_POP();
    return res;
}

// collect all of a method's live MethodInstances (the svec or dense form) —
// the conservative path pushes them all into oldmi, like activate's replaced
// branch does for the method it replaces.
static void membership_all_specializations(jl_method_t *m, jl_array_t *oldmi)
{
    jl_value_t *specs = (jl_value_t*)jl_atomic_load_relaxed(&m->specializations);
    _Atomic(jl_method_instance_t*) *data;
    size_t l;
    if (jl_is_svec(specs)) {
        data = (_Atomic(jl_method_instance_t*)*)jl_svec_data(specs);
        l = jl_svec_len(specs);
    }
    else {
        data = (_Atomic(jl_method_instance_t*)*)&m->specializations;
        l = 1;
    }
    for (size_t i = 0; i < l; i++) {
        jl_method_instance_t *mi = jl_atomic_load_relaxed(&data[i]);
        if ((jl_value_t*)mi == jl_nothing)
            continue;
        jl_array_ptr_1d_push(oldmi, (jl_value_t*)mi);
    }
}

// the conservative path has no substituted sig to consult invalidate_mt_cache's
// intersection/guardsig keeps against — a stale cache entry for a member call
// site need not intersect m's own sig at all — so every live mt-cache entry is
// lowered instead. Rare-shape-only; strictly more conservative than the walk.
static int invalidate_all_mt_cache(jl_typemap_entry_t *oldentry, void *closure0)
{
    struct invalidate_mt_env *env = (struct invalidate_mt_env*)closure0;
    if (jl_atomic_load_relaxed(&oldentry->max_world) == ~(size_t)0) {
        jl_atomic_store_relaxed(&oldentry->max_world, env->max_world);
        if (env->stats)
            env->stats->cache_lowerings++;
    }
    return 1;
}

// review W2-1 (second half): the conservative path's loser walk. A stale caller
// CI (or MethodInstance) decided pre-declare against m is invisible to the core
// invalidate (m has no specializations to walk) and unreachable by the
// substituted path's intersection machinery (there is no substituted sig): the
// specTypes that matter match m only once the membership is COMPOSED
// (fvt(::Any,::Any) at (typeof,MVt,MVt) vs (typeof,Vararg{VAt})). So ask every
// ground specialization of the table: if m's sig composed-matches (membership_-
// MATCH_SIG at the entry's world) where that spec was decided, its winner
// question changed — lower the CI chain (recursing into callers), clear the
// dispatch status, and shadow it. May lower specs another method still wins
// (recompute sooner) — conservative; rare shapes only.
struct membership_shadow_env {
    jl_method_t *m;          // the replayed set method
    size_t world;            // the entry's world (match composed at it)
    size_t max_world;        // last old world
    jl_array_t *oldmi;
    struct membership_stats *stats;
    int invalidated;
};
static int membership_shadow_visitor(jl_typemap_entry_t *def, void *closure0)
{
    struct membership_shadow_env *env = (struct membership_shadow_env*)closure0;
    // the driver keeps oldmi in its GC frame (gc[3]); the checker cannot see
    // through the void* closure — promise what is true (AGENTS.md convention)
    JL_GC_PROMISE_ROOTED(env->oldmi);
    jl_method_t *m2 = def->func.method;
    if (m2 == env->m || !jl_is_method((jl_value_t*)m2))
        return 1;
    // env->m is reachable from the globally rooted open_set_methods vec, and m2
    // through the walked typemap; root local copies for the GC-able calls below
    // (GCChecker cannot see either chain).
    jl_value_t *msig = (jl_value_t*)env->m->sig;
    jl_value_t *specs = (jl_value_t*)jl_atomic_load_relaxed(&m2->specializations);
    JL_GC_PUSH2(&msig, &specs);
    _Atomic(jl_method_instance_t*) *data;
    size_t l;
    if (jl_is_svec(specs)) {
        data = (_Atomic(jl_method_instance_t*)*)jl_svec_data(specs);
        l = jl_svec_len(specs);
    }
    else {
        data = (_Atomic(jl_method_instance_t*)*)&m2->specializations;
        l = 1;
    }
    for (size_t i = 0; i < l; i++) {
        jl_method_instance_t *mi = jl_atomic_load_relaxed(&data[i]);
        if ((jl_value_t*)mi == jl_nothing)
            continue;
        if (!jl_has_free_typevars((jl_value_t*)mi->specTypes) &&
            jl_membership_holds_at((jl_value_t*)mi->specTypes, msig, env->world)) {
            // mirroring the substituted path's replaced_dispatch arm: bound the
            // CI chain below the declaring world (recurses into the callers'
            // backedges), drop the dispatch fast path, shadow the MI.
            jl_code_instance_t *ci = membership_walk_invalidates ? jl_atomic_load_relaxed(&mi->cache) : NULL;
            while (ci) {
                invalidate_code_instance(ci, env->max_world, 1);
                ci = jl_atomic_load_relaxed(&ci->next);
            }
            jl_atomic_store_relaxed(&mi->dispatch_status, 0);
            jl_array_ptr_1d_push(env->oldmi, (jl_value_t*)mi);
            if (env->stats)
                env->stats->spec_invalidations++;
            env->invalidated = 1;
        }
    }
    JL_GC_POP();
    return 1;
}

// A method decides only the calls of its own function (compiler C-72): the walk's visits of
// the method table's definitions and of the method cache are by intersection with
// `Tuple{F, Vararg{Any}}`, F the method's first parameter, not over every function's entries
// (Julia's method table is one table for all functions). Exact: an entry another method's
// call reaches has a first parameter disjoint from F.
static jl_value_t *method_function_search_type(jl_method_t *m)
{
    jl_value_t *sig = (jl_value_t*)m->sig;
    jl_value_t *body = jl_unwrap_unionall(sig);
    if (!jl_is_tuple_type(body) || jl_nparams(body) == 0 || jl_is_vararg(jl_tparam0(body)))
        return (jl_value_t*)jl_anytuple_type;
    jl_value_t *params[2] = {jl_tparam0(body), jl_tparam0(jl_anytuple_type)}; // F, Vararg{Any}
    jl_value_t *t = (jl_value_t*)jl_apply_tuple_type_v(params, 2);
    JL_GC_PUSH1(&t);
    t = jl_rewrap_unionall(t, sig); // the method's variables that F names
    JL_GC_POP();
    return t;
}

struct membership_function_visit {
    struct typemap_intersection_env match;
    jl_typemap_visitor_fptr fptr;
    void *closure;
};
static int membership_function_visitor(jl_typemap_entry_t *e, struct typemap_intersection_env *closure0)
{
    struct membership_function_visit *env = container_of(closure0, struct membership_function_visit, match);
    return env->fptr(e, env->closure);
}
// calls fptr(entry, closure) on each entry of `map` whose signature meets `search`, live or not
// (the callbacks check the worlds themselves, as under jl_typemap_visitor)
static void membership_visit_function(jl_typemap_t *map, jl_value_t *search, size_t world,
                                      jl_typemap_visitor_fptr fptr, void *closure)
{
    jl_value_t *va = jl_unwrap_unionall(search);
    va = jl_tparam(va, jl_nparams(va) - 1);
    va = jl_is_vararg(va) ? jl_unwrap_vararg(va) : NULL;
    struct membership_function_visit env = {{membership_function_visitor, search, va,
            /* .search_slurp = */ 0, /* .world = */ world,
            /* .min_valid = */ 0, /* .max_valid = */ ~(size_t)0,
            /* .ti = */ NULL, /* .env = */ NULL, /* .issubty = */ 0},
        fptr, closure};
    JL_GC_PUSH1(&env.match.ti);
    jl_typemap_intersection_visitor(map, 0, &env.match);
    JL_GC_POP();
}

JL_DLLEXPORT void jl_membership_dispatch_walk(jl_value_t *member, jl_value_t *set_instance,
                                              size_t entry_min_world)
{
    if (!jl_atomic_load_relaxed(&allow_new_worlds))
        return; // entries land at package *execution*, not serialization (M6 replays them)
    size_t max_world = entry_min_world - 1; // I5: align to entry visibility (n3 semantics), not the counter
    // gc slots: 0 sets, 1 targets, 2 ms, 3 oldmi, 4 oldvalue, 5 dummy_sig, 6 dummy,
    // 7 loctag, 8 isect, 9 isect2. JL_GC_PUSHARGS slots hold the values themselves,
    // so the address-taking consumers below point at &gc[k] directly.
    jl_value_t **gc;
    JL_GC_PUSHARGS(gc, 10);
    membership_stats.declares++;

    jl_array_t *sets = (jl_array_t*)(gc[0] = (jl_value_t*)jl_alloc_vec_any(0));
    membership_superset_sets(set_instance, entry_min_world, sets);
    jl_array_t *targets = (jl_array_t*)(gc[1] = (jl_value_t*)jl_alloc_vec_any(0));
    for (size_t s = 0; s < jl_array_nrows(sets); s++) {
        jl_value_t *Yuw = jl_unwrap_unionall(jl_array_ptr_ref(sets, s));
        jl_array_t *ms = (jl_array_t*)(gc[2] = (jl_value_t*)jl_methods_at_set(((jl_datatype_t*)Yuw)->name));
        for (size_t i = 0; i < jl_array_nrows(ms); i++) {
            jl_value_t *m = jl_array_ptr_ref(ms, i);
            size_t k;
            for (k = 0; k < jl_array_nrows(targets); k++)
                if (jl_array_ptr_ref(targets, k) == m)
                    break;
            if (k == jl_array_nrows(targets))
                jl_array_ptr_1d_push(targets, m);
        }
    }
    gc[2] = NULL;

    for (size_t i = 0; i < jl_array_nrows(targets); i++) {
        jl_method_t *m = (jl_method_t*)jl_array_ptr_ref(targets, i);
        jl_methtable_t *mt = jl_method_get_table(m);
        if (!jl_is_mtable(mt))
            continue;
        int clean = 1;
        jl_value_t *substsing = membership_subst_sig(m, sets, member, &clean);
        if (clean == 2)
            continue; // m names the set only at other instances: the membership does not reach it
        membership_stats.replays++;
        int invalidated = 0;
        jl_value_t *type = NULL;
        jl_method_t *const *d = NULL;
        size_t j, n = 0;
        jl_value_t *search = gc[2] = method_function_search_type(m); // C-72: m's function only
        jl_value_t *dummy_sig = NULL;
        jl_typemap_entry_t *dummy = NULL;
        jl_array_t *oldmi = NULL;
        if (!clean || substsing == NULL) {
            // rare shape (set inside a parameter, or under Vararg — the latter
            // since review W2-5c made membership_subst_sig refuse vararg
            // substitutions): conservative full pass over m. No substituted sig
            // exists, so missing edges are pruned against m's own sig — with no
            // d[] to cover them, every intersecting edge is treated as missing
            // (never pruned). The invalidate CORE, not jl_method_table_invalidate:
            // the single-bump declare protocol holds world_counter_lock with the
            // counter already at min (review W2-3).
            if (membership_walk_invalidates)
                jl_method_invalidate_core(m, max_world, 0);
            oldmi = (jl_array_t*)(gc[3] = (jl_value_t*)jl_alloc_vec_any(0));
            membership_all_specializations(m, oldmi);
            struct membership_shadow_env shadow_env = {m, entry_min_world, max_world, oldmi,
                                                       &membership_stats, 0};
            membership_visit_function(jl_atomic_load_relaxed(&mt->defs), search, entry_min_world,
                                      membership_shadow_visitor, (void*)&shadow_env);
            invalidated |= shadow_env.invalidated;
            // a member's own methods now meet m's signature, so m is no longer the only method there;
            // `which(m.sig)` is still m (a membership adds no method): LATEST_WHICH stays, or validation
            // would read m as deleted and an image would drop its code
            jl_atomic_fetch_and_relaxed(&m->dispatch_status, ~METHOD_SIG_LATEST_ONLY);
            type = (jl_value_t*)m->sig;
        }
        else {
            dummy_sig = gc[5] = substsing;
            // never inserted; get_intersect_matches' asserts pass (min ~0, max 1).
            // this driver is the caller that roots it (review sweep 7).
            dummy = (jl_typemap_entry_t*)(gc[6] = (jl_value_t*)jl_typemap_alloc((jl_tupletype_t*)dummy_sig, NULL, jl_emptysvec, (jl_value_t*)m, ~(size_t)0, 1));
            jl_typemap_entry_t *replaced = NULL;
            gc[4] = get_intersect_matches(jl_atomic_load_relaxed(&mt->defs), dummy, &replaced, max_world, 1);
            assert(replaced == NULL); // a replay replaces nothing (its own entry is skipped, not equal)
            (void)replaced;
            if (gc[4] != NULL) {
                assert(jl_is_array(gc[4]));
                d = (jl_method_t**)jl_array_ptr_data((jl_array_t*)gc[4]);
                n = jl_array_nrows((jl_array_t*)gc[4]);
            }
            oldmi = (jl_array_t*)(gc[3] = (jl_value_t*)jl_alloc_vec_any(0));
            if (n > 0) {
                char *morespec = (char*)alloca(n);
                // M2a fix round (fix 4): the WINNER question compares the method's
                // ORIGINAL sig — the substituted dummy is trivially more specific
                // than every chain method (member <: chain type), which would let
                // the replay install member routing shadowed by a closed method
                // that C7 orders ABOVE the set. The substituted sig stays what
                // drives the intersections (which decisions are affected).
                for (j = 0; j < n; j++)
                    morespec[j] = (char)jl_type_morespecific(d[j]->sig, (jl_value_t*)m->sig);
                int dispatch_bits = 0; // replay discards these (activate-only publish)
                // M2a fix round (fix 4): the idset is seeded from the method's own
                // list, as activate's local is (gf.c:3343) — the has_key/put_key path
                // in invalidate_intersecting runs whenever a method is ordered ABOVE
                // the set method (morespec[j] = 1), which C7 now reaches at a replay;
                // the NULL seed segfaulted once fix 4 made that branch executable.
                // Published back below exactly as activate does: the additions are
                // what an insertion in this world would have produced, and without
                // the store the grown local copy would be a GC hazard.
                jl_genericmemory_t *new_interferences =
                    (jl_genericmemory_t*)jl_atomic_load_relaxed(&m->interferences);
                invalidated = invalidate_intersecting(dummy_sig, m, d, n, morespec, max_world,
                                                      oldmi, 1, &membership_stats, &dispatch_bits,
                                                      &new_interferences, &gc[8], &gc[9], &gc[7]);
                jl_atomic_store_release(&m->interferences, new_interferences);
                jl_gc_wb(m, new_interferences);
            }
            // clear m's fast path (probe 3 abused the stale LATEST_ONLY bit); LATEST_WHICH stays, as above
            jl_atomic_fetch_and_relaxed(&m->dispatch_status, ~METHOD_SIG_LATEST_ONLY);
            type = dummy_sig;
        }
        // tail exactly as activate
        jl_methcache_t *mc = jl_method_table->cache;
        JL_LOCK(&mc->writelock);
        // fix 4: intersections walk under the substituted sig, but the missing-edge
        // winner question answers with m's original sig (identical on the
        // conservative path, where type == m->sig already).
        struct _typename_invalidate_backedge typename_env = {type, (jl_value_t*)m->sig, &gc[8], &gc[9], d, n, max_world, invalidated, NULL};
        if (membership_walk_invalidates) {
            invalidate_missing_backedges(&typename_env);
            invalidated |= typename_env.invalidated;
        }
        // the cache walk needs the shadowed list on the substituted path; the
        // conservative path runs it unconditionally — its oldmi (all of m's own
        // specializations) can be legitimately EMPTY (a set method that was never
        // dispatched through) while stale mt/leaf entries for member call sites
        // exist for exactly the reasons invalidate_all_mt_cache documents (review
        // W2-1: this skip is what kept probe1's p2/p3 answering :old in the new
        // world after the indexing fix finally made the walk reach the method).
        if ((oldmi && jl_array_nrows(oldmi)) || !dummy) {
            struct invalidate_mt_env mt_cache_env;
            mt_cache_env.max_world = max_world;
            mt_cache_env.shadowed = oldmi;
            mt_cache_env.newentry = dummy;
            mt_cache_env.stats = &membership_stats;
            // C-72: the cache entries of m's function only; the leaf cache below is dropped whole, as at an insertion
            membership_visit_function(jl_atomic_load_relaxed(&mc->cache), search, entry_min_world,
                                      dummy ? invalidate_mt_cache : invalidate_all_mt_cache, (void*)&mt_cache_env);
            jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
            size_t li, ll = leafcache->length;
            for (li = 1; li < ll; li += 2) {
                jl_value_t *entry = jl_genericmemory_ptr_ref(leafcache, li);
                if (entry) {
                    while (entry != jl_nothing) {
                        jl_atomic_store_relaxed(&((jl_typemap_entry_t*)entry)->max_world, max_world);
                        entry = (jl_value_t*)jl_atomic_load_relaxed(&((jl_typemap_entry_t*)entry)->next);
                    }
                }
            }
            jl_atomic_store_relaxed(&mc->leafcache, (jl_genericmemory_t*)jl_an_empty_memory_any);
        }
        JL_UNLOCK(&mc->writelock);
        if (invalidated && _jl_debug_method_invalidation) {
            jl_array_ptr_1d_push(_jl_debug_method_invalidation, (jl_value_t*)m);
            gc[7] = (jl_value_t*)jl_cstr_to_string("jl_membership_dispatch_walk");
            jl_array_ptr_1d_push(_jl_debug_method_invalidation, gc[7]);
        }
        gc[2] = NULL;
        gc[3] = NULL;
        gc[4] = NULL;
        gc[5] = NULL;
        gc[6] = NULL;
    }
    jl_membership_log(member, set_instance, max_world);
    JL_GC_POP();
}

JL_DLLEXPORT void jl_method_table_insert(jl_methtable_t *mt, jl_method_t *method, jl_tupletype_t *simpletype)
{
    jl_typemap_entry_t *newentry = jl_method_table_add(mt, method, simpletype);
    JL_GC_PUSH1(&newentry);
    JL_LOCK(&world_counter_lock);
    if (!jl_atomic_load_relaxed(&allow_new_worlds))
        jl_error("Method changes have been disabled via a call to disable_new_worlds.");
    size_t world = jl_atomic_load_relaxed(&jl_world_counter) + 1;
    jl_atomic_store_relaxed(&method->primary_world, world);
    jl_method_table_activate(newentry);
    jl_atomic_store_release(&jl_world_counter, world);
    JL_UNLOCK(&world_counter_lock);
    JL_GC_POP();
}

static void JL_NORETURN jl_method_error_bare(jl_value_t *f, jl_value_t *args, size_t world)
{
    if (jl_methoderror_type) {
        jl_value_t *e = jl_new_struct_uninit(jl_methoderror_type);
        struct jl_method_error {
            jl_value_t *f;
            jl_value_t *args;
            size_t world;
        } *pe = (struct jl_method_error*)e,
           ee = {f, args, world};
        *pe = ee;
        jl_throw(e);
    }
    else {
        jl_printf((JL_STREAM*)STDERR_FILENO, "A method error occurred before the base MethodError type was defined. Aborting...\n");
        jl_static_show((JL_STREAM*)STDERR_FILENO,(jl_value_t*)f); jl_printf((JL_STREAM*)STDERR_FILENO," world %u\n", (unsigned)world);
        jl_static_show((JL_STREAM*)STDERR_FILENO,args); jl_printf((JL_STREAM*)STDERR_FILENO,"\n");
        jl_ptls_t ptls = jl_current_task->ptls;
        ptls->bt_size = rec_backtrace(ptls->bt_data, JL_MAX_BT_SIZE, 0);
        jl_fprint_critical_error(ios_safe_stderr, 0, 0, NULL, jl_current_task);
        abort();
    }
    // not reached
}

void JL_NORETURN jl_method_error(jl_value_t *f, jl_value_t **args, size_t na, size_t world)
{
    jl_value_t *argtup = jl_f_tuple(NULL, args, na - 1);
    JL_GC_PUSH1(&argtup);
    jl_method_error_bare(f, argtup, world);
    // not reached
}

jl_tupletype_t *arg_type_tuple(jl_value_t *arg1, jl_value_t **args, size_t nargs)
{
    return jl_inst_arg_tuple_type(arg1, args, nargs, 1);
}

static jl_tupletype_t *lookup_arg_type_tuple(jl_value_t *arg1 JL_PROPAGATES_ROOT, jl_value_t **args, size_t nargs)
{
    return jl_lookup_arg_tuple_type(arg1, args, nargs, 1);
}

JL_DLLEXPORT jl_value_t *jl_method_lookup_by_tt(jl_tupletype_t *tt, size_t world, jl_value_t *_mt)
{
    jl_methtable_t *mt = NULL;
    if (_mt == jl_nothing) {
        mt = jl_method_table;
    }
    else {
        assert(jl_is_mtable(_mt));
        mt = (jl_methtable_t*) _mt;
    }
    jl_methcache_t *mc = mt->cache;
    jl_method_instance_t *mi = jl_mt_assoc_by_type(mt, mc, tt, world);
    if (!mi)
        return jl_nothing;
    return (jl_value_t*) mi;
}

JL_DLLEXPORT jl_method_instance_t *jl_method_lookup(jl_value_t **args, size_t nargs, size_t world)
{
    assert(nargs > 0 && "expected caller to handle this case");
    jl_methcache_t *mc = jl_method_table->cache;
    jl_typemap_t *cache = jl_atomic_load_relaxed(&mc->cache); // XXX: gc root for this?
    jl_typemap_entry_t *entry = jl_typemap_assoc_exact(cache, args[0], &args[1], nargs, jl_cachearg_offset(), world);
    if (entry)
        return entry->func.linfo;
    jl_tupletype_t *tt = arg_type_tuple(args[0], &args[1], nargs);
    return jl_mt_assoc_by_type(jl_method_table, mc, tt, world);
}

// return a Vector{Any} of svecs, each describing a method match:
// Any[svec(tt, spvals, m, full), ...]
// tt is the intersection of the type argument and the method signature,
// spvals is any matched static parameter values, m is the Method,
// full is a boolean indicating if that method fully covers the input
//
// lim is the max # of methods to return. if there are more, returns jl_nothing.
// Negative values stand for no limit.
// Unless lim == -1, remove matches that are unambiguously covered by earlier ones
JL_DLLEXPORT jl_value_t *jl_matching_methods(jl_tupletype_t *types, jl_value_t *mt, int lim, int include_ambiguous,
                                             size_t world, size_t *min_valid, size_t *max_valid, int *ambig)
{
    if (ambig != NULL)
        *ambig = 0;
    jl_value_t *unw = jl_unwrap_unionall((jl_value_t*)types);
    if (!jl_is_tuple_type(unw))
        return (jl_value_t*)jl_an_empty_vec_any;
    if (unw == (jl_value_t*)jl_emptytuple_type || jl_tparam0(unw) == jl_bottom_type)
        return (jl_value_t*)jl_an_empty_vec_any;
    if (mt == jl_nothing)
        mt = (jl_value_t*)jl_method_table;
    jl_methcache_t *mc = ((jl_methtable_t*)mt)->cache;
    return ml_matches((jl_methtable_t*)mt, mc, types, lim, include_ambiguous, 1, world, 1, min_valid, max_valid, ambig);
}

JL_DLLEXPORT jl_method_instance_t *jl_get_unspecialized(jl_method_t *def JL_PROPAGATES_ROOT)
{
    // one unspecialized version of a function can be shared among all cached specializations
    if (!jl_is_method(def) || def->source == NULL) {
        // generated functions might instead randomly just never get inferred, sorry
        return (jl_method_instance_t*)jl_nothing;
    }
    jl_method_instance_t *unspec = jl_atomic_load_relaxed(&def->unspecialized);
    if (unspec == NULL) {
        JL_LOCK(&def->writelock);
        unspec = jl_atomic_load_relaxed(&def->unspecialized);
        if (unspec == NULL) {
            unspec = jl_get_specialized(def, def->sig, jl_emptysvec);
            jl_atomic_store_release(&def->unspecialized, unspec);
            jl_gc_wb(def, unspec);
        }
        JL_UNLOCK(&def->writelock);
    }
    return unspec;
}

STATIC_INLINE jl_value_t *_jl_rettype_inferred(jl_value_t *owner, jl_method_instance_t *mi, size_t min_world, size_t max_world, int facts) JL_NOTSAFEPOINT
{
    jl_code_instance_t *codeinst = jl_atomic_load_relaxed(&mi->cache);
    while (codeinst) {
        if (jl_atomic_load_relaxed(&codeinst->min_world) <= min_world &&
            max_world <= jl_atomic_load_relaxed(&codeinst->max_world) &&
            jl_egal(codeinst->owner, owner) &&
            (owner != jl_nothing || ci_policy_is(codeinst, facts))) {

            jl_value_t *code = jl_atomic_load_relaxed(&codeinst->inferred);
            if (code)
                return (jl_value_t*)codeinst;
        }
        codeinst = jl_atomic_load_relaxed(&codeinst->next);
    }
    return (jl_value_t*)jl_nothing;
}

JL_DLLEXPORT jl_value_t *jl_rettype_inferred(jl_value_t *owner, jl_method_instance_t *mi, size_t min_world, size_t max_world) JL_NOTSAFEPOINT
{
    return (jl_value_t*)_jl_rettype_inferred(owner, mi, min_world, max_world, mi_policy_facts_at(mi, JL_REACH_RUNTIME));
}

// the same, for code of a policy (compiler §2.3): `policy` is 0 plain, 1 facts, or -1 for the policy
// the runtime reaches it with; a method of a policy of its own has code of that policy only
JL_DLLEXPORT jl_value_t *jl_rettype_inferred_policy(jl_value_t *owner, jl_method_instance_t *mi, size_t min_world, size_t max_world, int policy) JL_NOTSAFEPOINT
{
    int facts = mi_policy_facts_at(mi, policy < 0 ? JL_REACH_RUNTIME : policy ? JL_REACH_FACTS : JL_REACH_PLAIN);
    return (jl_value_t*)_jl_rettype_inferred(owner, mi, min_world, max_world, facts);
}

JL_DLLEXPORT jl_value_t *jl_rettype_inferred_native(jl_method_instance_t *mi, size_t min_world, size_t max_world) JL_NOTSAFEPOINT
{
    return (jl_value_t*)_jl_rettype_inferred(jl_nothing, mi, min_world, max_world, mi_policy_facts_at(mi, JL_REACH_RUNTIME));
}

JL_DLLEXPORT jl_value_t *(*const jl_rettype_inferred_addr)(jl_method_instance_t *mi, size_t min_world, size_t max_world) JL_NOTSAFEPOINT = jl_rettype_inferred_native;

STATIC_INLINE jl_callptr_t jl_method_compiled_callptr(jl_method_instance_t *mi, size_t world, int facts, jl_code_instance_t **codeinst_out) JL_NOTSAFEPOINT
{
    jl_code_instance_t *codeinst = jl_atomic_load_relaxed(&mi->cache);
    for (; codeinst; codeinst = jl_atomic_load_relaxed(&codeinst->next)) {
        if (codeinst->owner != jl_nothing || !ci_policy_is(codeinst, facts))
            continue;
        if (jl_atomic_load_relaxed(&codeinst->min_world) <= world && world <= jl_atomic_load_relaxed(&codeinst->max_world)) {
            jl_callptr_t invoke = jl_atomic_load_acquire(&codeinst->invoke);
            if (!invoke)
                continue;
            *codeinst_out = codeinst;
            return invoke;
        }
    }
    return NULL;
}

jl_code_instance_t *jl_method_compiled(jl_method_instance_t *mi, size_t world) JL_NOTSAFEPOINT
{
    return jl_method_compiled_policy(mi, world, mi_policy_facts_at(mi, JL_REACH_RUNTIME));
}

jl_code_instance_t *jl_method_compiled_policy(jl_method_instance_t *mi, size_t world, int facts) JL_NOTSAFEPOINT
{
    jl_code_instance_t *codeinst = NULL;
    jl_method_compiled_callptr(mi, world, facts, &codeinst);
    return codeinst;
}

jl_mutex_t precomp_statement_out_lock;

_Atomic(uint8_t) jl_force_trace_compile_timing_enabled = 0;

/**
 * @brief Enable force trace compile to stderr with timing.
 */
JL_DLLEXPORT void jl_force_trace_compile_timing_enable(void)
{
    // Increment the flag to allow reentrant callers to `@trace_compile`.
    jl_atomic_fetch_add(&jl_force_trace_compile_timing_enabled, 1);
}
/**
 * @brief Disable force trace compile to stderr with timing.
 */
JL_DLLEXPORT void jl_force_trace_compile_timing_disable(void)
{
    // Increment the flag to allow reentrant callers to `@trace_compile`.
    jl_atomic_fetch_add(&jl_force_trace_compile_timing_enabled, -1);
}

static void record_precompile_statement(jl_method_instance_t *mi, double compilation_time, int is_recompile)
{
    static ios_t f_precompile;
    static JL_STREAM* s_precompile = NULL;
    jl_method_t *def = mi->def.method;
    uint8_t force_trace_compile = jl_atomic_load_relaxed(&jl_force_trace_compile_timing_enabled);
    if (force_trace_compile == 0 && jl_options.trace_compile == NULL)
        return;
    if (!jl_is_method(def))
        return;
    if (def->is_for_opaque_closure)
        return; // OpaqueClosure methods cannot be looked up by their types, so are incompatible with `precompile(...)`

    JL_LOCK(&precomp_statement_out_lock);
    if (s_precompile == NULL) {
        const char *t = jl_options.trace_compile;
        if (force_trace_compile || !strncmp(t, "stderr", 6)) {
            s_precompile = JL_STDERR;
        }
        else {
            if (ios_file(&f_precompile, t, 1, 1, 1, 1) == NULL)
                jl_errorf("cannot open precompile statement file \"%s\" for writing", t);
            s_precompile = (JL_STREAM*) &f_precompile;
        }
    }
    if (!jl_has_free_typevars(mi->specTypes)) {
        if (is_recompile && s_precompile == JL_STDERR && jl_options.color != JL_OPTIONS_COLOR_OFF)
            jl_printf(s_precompile, "\e[33m");
        if (force_trace_compile || jl_options.trace_compile_timing)
            jl_printf(s_precompile, "#= %6.1f ms =# ", compilation_time / 1e6);
        jl_printf(s_precompile, "precompile(");
        jl_static_show(s_precompile, mi->specTypes);
        jl_printf(s_precompile, ")");
        if (is_recompile) {
            jl_printf(s_precompile, " # recompile");
            if (s_precompile == JL_STDERR && jl_options.color != JL_OPTIONS_COLOR_OFF) {
                jl_printf(s_precompile, "\e[0m");
            }
        }
        jl_printf(s_precompile, "\n");
        if (s_precompile != JL_STDERR)
            ios_flush(&f_precompile);
    }
    JL_UNLOCK(&precomp_statement_out_lock);
}

jl_mutex_t dispatch_statement_out_lock;

_Atomic(uint8_t) jl_force_trace_dispatch_enabled = 0;

/**
 * @brief Enable force trace dispatch to stderr.
 */
JL_DLLEXPORT void jl_force_trace_dispatch_enable(void)
{
    // Increment the flag to allow reentrant callers to `@trace_dispatch`.
    jl_atomic_fetch_add(&jl_force_trace_dispatch_enabled, 1);
}
/**
 * @brief Disable force trace dispatch to stderr.
 */
JL_DLLEXPORT void jl_force_trace_dispatch_disable(void)
{
    // Increment the flag to allow reentrant callers to `@trace_dispatch`.
    jl_atomic_fetch_add(&jl_force_trace_dispatch_enabled, -1);
}

static void record_dispatch_statement(jl_method_instance_t *mi)
{
    static ios_t f_dispatch;
    static JL_STREAM* s_dispatch = NULL;
    jl_method_t *def = mi->def.method;
    if (!jl_is_method(def))
        return;

    uint8_t force_trace_dispatch = jl_atomic_load_relaxed(&jl_force_trace_dispatch_enabled);
    JL_LOCK(&dispatch_statement_out_lock);
    if (s_dispatch == NULL) {
        const char *t = jl_options.trace_dispatch;
        if (force_trace_dispatch || !strncmp(t, "stderr", 6)) {
            s_dispatch = JL_STDERR;
        }
        else {
            if (ios_file(&f_dispatch, t, 1, 1, 1, 1) == NULL)
                jl_errorf("cannot open dispatch statement file \"%s\" for writing", t);
            s_dispatch = (JL_STREAM*) &f_dispatch;
        }
    }
    // NOTE: For builtin functions, the specType is just `Tuple`, which is not useful to print.
    if (!jl_has_free_typevars(mi->specTypes) && (jl_datatype_t*)mi->specTypes != jl_tuple_type) {
        jl_printf(s_dispatch, "precompile(");
        jl_static_show(s_dispatch, mi->specTypes);
        jl_printf(s_dispatch, ")\n");
        if (s_dispatch != JL_STDERR)
            ios_flush(&f_dispatch);
    }
    JL_UNLOCK(&dispatch_statement_out_lock);
}

static void record_dispatch_statement_on_first_dispatch(jl_method_instance_t *mfunc) {
    uint8_t force_trace_dispatch = jl_atomic_load_relaxed(&jl_force_trace_dispatch_enabled);
    if (force_trace_dispatch || jl_options.trace_dispatch != NULL) {
        uint8_t miflags = jl_atomic_load_relaxed(&mfunc->flags);
        uint8_t was_dispatched = miflags & JL_MI_FLAGS_MASK_DISPATCHED;
        if (!was_dispatched) {
            miflags |= JL_MI_FLAGS_MASK_DISPATCHED;
            jl_atomic_store_relaxed(&mfunc->flags, miflags);
            record_dispatch_statement(mfunc);
        }
    }
}

// If waitcompile is 0, this will return NULL if compiling is on-going in the JIT. This is
// useful for the JIT itself, since it just doesn't cause redundant work or missed updates,
// but merely causes it to look into the current JIT worklist.
void jl_read_codeinst_invoke(jl_code_instance_t *ci, uint8_t *specsigflags, jl_callptr_t *invoke, void **specptr, int waitcompile)
{
    uint8_t flags = jl_atomic_load_acquire(&ci->flags); // happens-before for subsequent read of fptr
    while (1) {
        jl_callptr_t initial_invoke = jl_atomic_load_acquire(&ci->invoke); // happens-before for subsequent read of fptr
        if (initial_invoke == jl_fptr_wait_for_compiled_addr) {
            if (!waitcompile) {
                *invoke = NULL;
                *specptr = NULL;
                *specsigflags = 0b00;
                return;
            }
            jl_compile_codeinst(ci);
            initial_invoke = jl_atomic_load_acquire(&ci->invoke); // happens-before for subsequent read of fptr
        }
        void *fptr = jl_atomic_load_relaxed(&ci->specptr.fptr);
        // TODO: if fptr is NULL, it may mean we read this too fast, and should have spun and waited for jl_compile_codeinst to finish
        if (initial_invoke == NULL || fptr == NULL) {
            *invoke = initial_invoke;
            *specptr = NULL;
            *specsigflags = 0b00;
            return;
        }
        while (!(flags & JL_CI_FLAGS_INVOKE_MATCHES_SPECPTR)) {
            jl_cpu_pause();
            flags = jl_atomic_load_acquire(&ci->flags);
        }
        jl_callptr_t final_invoke = jl_atomic_load_relaxed(&ci->invoke);
        if (final_invoke == initial_invoke) {
            *invoke = final_invoke;
            *specptr = fptr;
            *specsigflags = flags;
            return;
        }
    }
}

// The specialization a static call with argument types `tt` runs at `m` (compiler §6.3): the declared
// type at each trait argument whose attribute is `Dynamic`, the cast or carried attributes `mask` and
// `dyn` first, the rest resolved at the method. Inference infers the call there, so that what it
// infers is the instance that runs. `tt` itself where nothing is compiled once.
JL_DLLEXPORT jl_value_t *jl_attribute_sig(jl_value_t *tt, jl_method_t *m, uint32_t mask, uint32_t dyn)
{
    if (!jl_is_datatype(tt) || !jl_is_tuple_type(tt))
        return tt;
    uint32_t traits = trait_arg_bits(m);
    if (traits == 0)
        return tt;
    uint32_t nospec = effective_nospecialize(m, (jl_tupletype_t*)tt, 1, mask, dyn) & traits;
    if (nospec == 0)
        return tt;
    jl_svec_t *params = NULL, *env = NULL;
    jl_value_t *decl_i = NULL;
    JL_GC_PUSH4(&params, &tt, &env, &decl_i);
    size_t np = jl_nparams(tt);
    for (size_t i = 1; i < np && i <= 32; i++) {
        if (!(nospec & (1u << (i - 1))))
            continue;
        decl_i = jl_nth_slot_type(m->sig, i);
        if (jl_has_free_typevars(decl_i)) {
            // a declared type that names the method's variables (`X{T}`, C-54): its instance with their
            // values from the call, or the type over them where the call does not fix them
            if (env == NULL) {
                env = jl_emptysvec;
                if (jl_is_unionall(m->sig) && jl_type_intersection_env(tt, m->sig, &env) == jl_bottom_type)
                    env = jl_emptysvec;
            }
            jl_value_t *inst = NULL;
            if (jl_is_unionall(m->sig) && jl_svec_len(env) > 0)
                inst = jl_instantiate_type_in_env(decl_i, (jl_unionall_t*)m->sig, jl_svec_data(env));
            decl_i = (inst == NULL || jl_has_free_typevars(inst)) ? jl_rewrap_unionall(decl_i, m->sig) : inst;
        }
        if (jl_egal(jl_tparam(tt, i), decl_i))
            continue;
        if (params == NULL)
            params = jl_svec_copy(((jl_datatype_t*)tt)->parameters);
        jl_svecset(params, i, decl_i);
    }
    if (params != NULL)
        tt = (jl_value_t*)jl_apply_tuple_type(params, 1);
    JL_GC_POP();
    return tt;
}

// The trait arguments of `m` that a static call with argument types `tt` compiles once (compiler
// §6.3), with the cast or carried attributes `mask` and `dyn` first.
JL_DLLEXPORT uint32_t jl_attribute_dynamic_bits(jl_value_t *tt, jl_method_t *m, uint32_t mask, uint32_t dyn)
{
    if (!jl_is_datatype(tt) || !jl_is_tuple_type(tt))
        return 0;
    uint32_t traits = trait_arg_bits(m);
    return traits ? effective_nospecialize(m, (jl_tupletype_t*)tt, 1, mask, dyn) & traits : 0;
}
JL_DLLEXPORT uint32_t jl_trait_arg_bits(jl_method_t *m)
{
    return trait_arg_bits(m);
}

// The method instance a call with argument types `tt` runs with the attribute bits `mask` (the
// trait arguments decided) and `dyn` (those of them that are `Dynamic`): the method dispatch selects,
// compiled with those attributes and the rest resolved at the method for a dynamic call (compiler
// §6.3, §6.9). NULL where dispatch selects no single method: the caller dispatches as usual then.
JL_DLLEXPORT jl_method_instance_t *jl_attributed_mi(jl_tupletype_t *tt, uint32_t mask, uint32_t dyn, size_t world)
{
    size_t min_valid = 1, max_valid = ~(size_t)0;
    jl_value_t *matches = jl_matching_methods(tt, jl_nothing, 1, 0, world, &min_valid, &max_valid, NULL);
    if (!jl_is_array(matches) || jl_array_nrows((jl_array_t*)matches) != 1)
        return NULL;
    jl_method_match_t *match = (jl_method_match_t*)jl_array_ptr_ref((jl_array_t*)matches, 0);
    if (!match->fully_covers)
        return NULL;
    jl_method_t *m = match->method;
    jl_svec_t *env = match->sparams;
    jl_tupletype_t *ti = match->spec_types;
    jl_svec_t *newparams = NULL;
    jl_value_t *sig = NULL;
    jl_method_instance_t *mi = NULL;
    JL_GC_PUSH5(&matches, &env, &newparams, &sig, &mi);
    jl_compilation_sig(ti, env, m, get_max_varargs(m, NULL),
                       effective_nospecialize(m, ti, 0, mask, dyn), &newparams);
    if (newparams) {
        sig = (jl_value_t*)jl_apply_tuple_type(newparams, 1);
        newparams = NULL;
        jl_value_t *ti2 = jl_type_intersection_env(sig, m->sig, &newparams);
        if (ti2 != jl_bottom_type) {
            env = newparams;
            ti = (jl_tupletype_t*)sig;
        }
    }
    mi = jl_specializations_get_linfo(m, (jl_value_t*)ti, env);
    JL_GC_POP();
    return mi;
}

jl_method_instance_t *jl_normalize_to_compilable_mi(jl_method_instance_t *mi JL_PROPAGATES_ROOT);

JL_DLLEXPORT void jl_add_codeinst_to_jit(jl_code_instance_t *codeinst, jl_code_info_t *src)
{
    assert(jl_is_code_info(src));
    jl_emit_codeinst_to_jit(codeinst, src);
}

JL_DLLEXPORT int jl_method_is_macro(jl_method_t *m)
{
    return jl_symbol_name(m->name)[0] == '@';
}

// ---- The code slot's revalidation (compiler §3.3) ----

// two result types the same type: egal, or equal as types (a `where` clause's variables may be
// named differently in two CodeInstances)
static int same_type(jl_value_t *a, jl_value_t *b)
{
    if (a == b || jl_egal(a, b))
        return 1;
    return jl_is_type(a) && jl_is_type(b) && jl_types_equal(a, b);
}

// whether a callee's new CodeInstance has the result its caller used: return type, exception type,
// constant and effects
static int ci_result_equal(jl_code_instance_t *a, jl_code_instance_t *b)
{
    if (!same_type(a->rettype, b->rettype) || !same_type(a->exctype, b->exctype))
        return 0;
    if ((a->rettype_const == NULL) != (b->rettype_const == NULL))
        return 0;
    if (a->rettype_const != NULL && !jl_egal(a->rettype_const, b->rettype_const))
        return 0;
    if ((jl_atomic_load_relaxed(&a->invoke) == jl_fptr_const_return) != (jl_atomic_load_relaxed(&b->invoke) == jl_fptr_const_return))
        return 0;
    return jl_atomic_load_relaxed(&a->ipo_purity_bits) == jl_atomic_load_relaxed(&b->ipo_purity_bits);
}

static int revalidate_code_instance(jl_code_instance_t *ci, size_t world, arraylist_t *stack);

static int ci_valid_in(jl_code_instance_t *ci, size_t world, arraylist_t *stack)
{
    if (jl_atomic_load_acquire(&ci->max_world) >= world)
        return jl_atomic_load_relaxed(&ci->min_world) <= world;
    return revalidate_code_instance(ci, world, stack);
}

static int edge_is_single(jl_value_t *e) JL_NOTSAFEPOINT
{
    return jl_is_method_instance(e) || jl_is_method(e) || jl_is_binding(e) ||
        jl_typetagis(e, jl_membership_cell_type) || jl_typetagis(e, jl_membership_type);
}

// Revalidates a marked CodeInstance in the latest world `world`, depth first; a cycle fails. Each of
// its edges to a CodeInstance not valid there: a marked one is revalidated first; for a result edge,
// the callee is inferred again and must have an equal result, valid from the world of the mark on.
// Every other edge is unchanged since the mark, or the mark would have become final. On success the
// edges name the callees' new CodeInstances, and max_world is restored; on failure the mark becomes
// an invalidation.
static int revalidate_code_instance(jl_code_instance_t *ci, size_t world, arraylist_t *stack)
{
    size_t maxw = jl_atomic_load_acquire(&ci->max_world);
    if (maxw >= world)
        return 1;
    if (!(jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_MARKED))
        return 0;
    size_t markworld = maxw + 1;
    for (size_t k = 0; k < stack->len; k++)
        if (stack->items[k] == (void*)ci)
            return 0;
    arraylist_push(stack, (void*)ci);
    jl_svec_t *edges = jl_atomic_load_acquire(&ci->edges);
    jl_svec_t *redges = ci_result_edges(ci);
    jl_svec_t *newedges = NULL;
    jl_code_instance_t *e2 = NULL;
    JL_GC_PUSH4(&edges, &redges, &newedges, &e2);
    int ok = 1;
    size_t n = jl_svec_len(edges), i = 0;
    while (ok && i < n) {
        jl_value_t *e = jl_svecref(edges, i);
        if (jl_is_long(e)) {            // a match list: count, signature, its targets
            long k = jl_unbox_long(e);
            size_t m = (size_t)(k < 0 ? -k : k);
            for (size_t j = i + 2; ok && j < i + 2 + m && j < n; j++) {
                jl_value_t *x = jl_svecref(edges, j);
                if (jl_is_code_instance(x) && !ci_valid_in((jl_code_instance_t*)x, world, stack))
                    ok = 0;
            }
            i += 2 + m;
            continue;
        }
        if (jl_is_code_instance(e)) {
            jl_code_instance_t *ce = (jl_code_instance_t*)e;
            if (!ci_valid_in(ce, world, stack)) {
                jl_method_instance_t *emi = jl_get_ci_mi(ce);
                int is_result = 0;
                for (size_t j = 0; redges != NULL && j < jl_svec_len(redges); j++)
                    is_result |= jl_svecref(redges, j) == (jl_value_t*)emi;
                if (!is_result || ce->owner != jl_nothing) {
                    ok = 0;
                    break;
                }
                e2 = jl_type_infer_policy(emi, world, SOURCE_MODE_NOT_REQUIRED, jl_options.trim, ci_policy_facts(ce));
                if (e2 == NULL || !ci_result_equal(ce, e2) ||
                        jl_atomic_load_relaxed(&e2->min_world) > markworld ||
                        jl_atomic_load_relaxed(&e2->max_world) < world) {
                    ok = 0;
                    break;
                }
                if (newedges == NULL)
                    newedges = jl_svec_copy(edges);
                jl_svecset(newedges, i, (jl_value_t*)e2);
            }
            i++;
            continue;
        }
        if (edge_is_single(e)) {
            i++;
            continue;
        }
        // a pair: an `invoke` edge, or a signature and its method table
        if (i + 1 < n) {
            jl_value_t *x = jl_svecref(edges, i + 1);
            if (jl_is_code_instance(x) && !ci_valid_in((jl_code_instance_t*)x, world, stack))
                ok = 0;
        }
        i += 2;
    }
    arraylist_pop(stack);
    // commit in the world the comparison was made in
    jl_method_t *m = jl_get_ci_mi(ci)->def.method;
    JL_LOCK(&world_counter_lock);
    int same_world = jl_atomic_load_relaxed(&jl_world_counter) == world;
    if (jl_is_method(m))
        JL_LOCK(&m->writelock);
    int still_marked = (jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_MARKED) &&
        jl_atomic_load_relaxed(&ci->max_world) == maxw;
    // every callee CodeInstance the revalidation accepted is still valid here: an invalidation
    // without a new world (image validation on another thread) may have come since the walk
    if (ok && same_world && still_marked) {
        jl_svec_t *final = newedges != NULL ? newedges : edges;
        for (size_t j = 0; ok && j < jl_svec_len(final); j++) {
            jl_value_t *x = jl_svecref(final, j);
            if (jl_is_code_instance(x) && x != (jl_value_t*)ci &&
                    jl_atomic_load_relaxed(&((jl_code_instance_t*)x)->max_world) < world)
                ok = 0;
        }
    }
    if (ok && same_world && still_marked) {
        if (newedges != NULL) {
            jl_atomic_store_release(&ci->edges, newedges);
            jl_gc_wb(ci, newedges);
        }
        jl_atomic_fetch_and_relaxed(&ci->flags, (uint8_t)~JL_CI_FLAGS_MARKED);
        jl_atomic_store_release(&ci->max_world, ~(size_t)0);
        CODE_SLOT_COUNT(kept);
    }
    else {
        if (same_world && still_marked) {
            jl_atomic_fetch_and_relaxed(&ci->flags, (uint8_t)~JL_CI_FLAGS_MARKED);
            CODE_SLOT_COUNT(final_marks);
        }
        ok = 0;
    }
    if (jl_is_method(m))
        JL_UNLOCK(&m->writelock);
    JL_UNLOCK(&world_counter_lock);
    JL_GC_POP();
    return ok;
}

// a marked CodeInstance of `mi` and the policy (compiled, where `compiled`) made valid again in the
// latest world
static jl_code_instance_t *revalidate_marked(jl_method_instance_t *mi JL_PROPAGATES_ROOT, size_t world, int facts, int compiled)
{
    if (!jl_code_slot_on() || world != jl_atomic_load_acquire(&jl_world_counter))
        return NULL;
    jl_code_instance_t *ci = jl_atomic_load_relaxed(&mi->cache);
    for (; ci; ci = jl_atomic_load_relaxed(&ci->next)) {
        if (ci->owner != jl_nothing || !ci_policy_is(ci, facts) ||
                !(jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_MARKED) ||
                (compiled && jl_atomic_load_relaxed(&ci->invoke) == NULL) ||
                jl_atomic_load_relaxed(&ci->inferred) == NULL ||
                jl_atomic_load_relaxed(&ci->min_world) > world)
            continue;
        arraylist_t stack;
        arraylist_new(&stack, 0);
        int ok = revalidate_code_instance(ci, world, &stack);
        arraylist_free(&stack);
        if (ok)
            return ci;
    }
    return NULL;
}

static jl_code_instance_t *jl_revalidate_policy(jl_method_instance_t *mi JL_PROPAGATES_ROOT, size_t world, int facts)
{
    return revalidate_marked(mi, world, facts, 1);
}

// inference's cache miss (compiler §3.3): a marked CodeInstance made valid again serves as the
// cached result, so code that calls it does not infer it again
// nonzero while this thread revalidates: a revalidation's own inference infers afresh
static _Thread_local int code_slot_revalidating = 0;

JL_DLLEXPORT jl_value_t *jl_revalidate_for_inference(jl_method_instance_t *mi, size_t world, int facts)
{
    if (code_slot_revalidating)
        return jl_nothing;
    code_slot_revalidating = 1;
    jl_code_instance_t *ci = revalidate_marked(mi, world, facts, 0);
    code_slot_revalidating = 0;
    return ci != NULL ? (jl_value_t*)ci : jl_nothing;
}

// whether the CodeInstance has the entry a slot calls: the specialized signature, or the generic one
static int code_slot_entry(jl_code_instance_t *ci, int specsig)
{
    uint8_t specsigflags;
    jl_callptr_t invoke;
    void *fptr;
    jl_read_codeinst_invoke(ci, &specsigflags, &invoke, &fptr, 1);
    if (invoke == NULL || invoke == jl_fptr_const_return)
        return 0;
    return specsig ? (specsigflags & JL_CI_FLAGS_SPECPTR_SPECIALIZED) && fptr != NULL : 1;
}

// a CodeInstance of `mi` and the policy, valid in `world`, with the result type the caller was compiled
// against and the entry its call needs
static jl_code_instance_t *code_slot_find(jl_method_instance_t *mi JL_PROPAGATES_ROOT, size_t world, int facts, jl_value_t *rettype, int specsig)
{
    jl_code_instance_t *ci = jl_atomic_load_relaxed(&mi->cache);
    for (; ci; ci = jl_atomic_load_relaxed(&ci->next)) {
        if (ci->owner == jl_nothing && ci_policy_is(ci, facts) &&
                jl_atomic_load_relaxed(&ci->min_world) <= world && world <= jl_atomic_load_relaxed(&ci->max_world) &&
                same_type(ci->rettype, rettype) && code_slot_entry(ci, specsig))
            return ci;
    }
    return NULL;
}

static void code_slot_show(jl_code_instance_t *ci) JL_NOTSAFEPOINT
{
    jl_printf(JL_STDERR, "  CodeInstance worlds [%zu, %zu] flags 0x%x invoke %p specptr %p rettype ",
              jl_atomic_load_relaxed(&ci->min_world), jl_atomic_load_relaxed(&ci->max_world),
              (unsigned)jl_atomic_load_relaxed(&ci->flags), (void*)jl_atomic_load_relaxed(&ci->invoke),
              jl_atomic_load_relaxed(&ci->specptr.fptr));
    jl_static_show(JL_STDERR, ci->rettype);
    jl_printf(JL_STDERR, "\n");
}

// The slot's last resort (compiler §3.3): where no code of `mi` with the caller's result type and the
// entry its call needs can be made in the task's world, an entry with the call's signature that
// dispatches when the call runs, in the task's world, and asserts the caller's result type (the ABI
// adapter's conversion from `Any`). A dynamic call at the static call's signature selects the
// caller's method wherever the caller is valid. A boxed call runs the method's code in the
// interpreter. It is held by a CodeInstance of its own, valid in every world, kept for the session
// (the slot keeps a raw pointer to it).
jl_array_t *jl_code_slot_fallbacks JL_GLOBALLY_ROOTED = NULL;   // a GC root (gc_mark_roots)
static jl_mutex_t code_slot_fallback_lock;
// for tests: every miss takes the last resort
static int code_slot_force_fallback = 0;
JL_DLLEXPORT int jl_code_slot_force_fallback(int on) JL_NOTSAFEPOINT
{
    int last = code_slot_force_fallback;
    code_slot_force_fallback = on;
    return last;
}

static jl_code_instance_t *code_slot_fallback(jl_method_instance_t *mi, jl_value_t *rettype, int specsig)
{
    jl_code_instance_t *ci = NULL;
    JL_LOCK(&code_slot_fallback_lock);
    if (jl_code_slot_fallbacks != NULL) {
        for (size_t i = 0; i < jl_array_nrows(jl_code_slot_fallbacks); i++) {
            jl_code_instance_t *c = (jl_code_instance_t*)jl_array_ptr_ref(jl_code_slot_fallbacks, i);
            if (jl_get_ci_mi(c) == mi && same_type(c->rettype, rettype)) {
                ci = c;
                break;
            }
        }
    }
    JL_UNLOCK(&code_slot_fallback_lock);
    if (ci != NULL)
        return ci;
    void *f = NULL;
    if (specsig) {
        jl_abi_t abi = { mi->specTypes, rettype, jl_nparams(mi->specTypes), 1, 0 };
        f = jl_jit_abi_converter(jl_current_task, abi, NULL);
        if (f == NULL)
            return NULL;
    }
    ci = jl_new_codeinst(mi, jl_nothing, rettype, (jl_value_t*)jl_any_type, NULL, NULL,
                         0, 1, ~(size_t)0, 0, jl_nothing, NULL, jl_emptysvec);
    JL_GC_PUSH1(&ci);
    if (specsig) {
        jl_atomic_store_relaxed(&ci->specptr.fptr, f);
        jl_atomic_store_relaxed(&ci->flags, JL_CI_FLAGS_SPECPTR_SPECIALIZED);
    }
    // the generic entry runs the method's code in the interpreter (a boxed call's last resort)
    jl_atomic_store_release(&ci->invoke, jl_fptr_interpret_call);
    JL_LOCK(&code_slot_fallback_lock);
    if (jl_code_slot_fallbacks == NULL)
        jl_code_slot_fallbacks = jl_alloc_vec_any(0);
    jl_array_ptr_1d_push(jl_code_slot_fallbacks, (jl_value_t*)ci);
    JL_UNLOCK(&code_slot_fallback_lock);
    JL_GC_POP();
    return ci;
}

// A slot's miss (compiler §3.3): the code of `mi` and the policy valid in `world`, revalidated or
// compiled as needed, with the result type the caller was compiled against, put in the slot.
// `target` is the CodeInstance the call was compiled against, where codegen knows it (the JIT).
JL_DLLEXPORT jl_code_instance_t *jl_code_slot_resolve(_Atomic(jl_code_instance_t*) *slot, jl_method_instance_t *mi, jl_value_t *rettype, size_t world, uint32_t flags, jl_code_instance_t *target)
{
    int specsig = flags & 1;
    int facts = (flags & 2) != 0;
    CODE_SLOT_COUNT(misses);
    jl_code_instance_t *ci = NULL;
    if (target != NULL && jl_atomic_load_relaxed(&target->min_world) <= world &&
            world <= jl_atomic_load_relaxed(&target->max_world)) {
        // the code the call was compiled against, valid here: compiled now if the JIT has it only
        // emitted (a slot links nothing to it)
        if (!jl_is_compiled_codeinst(target))
            jl_compile_codeinst(target);
        if (code_slot_entry(target, specsig))
            ci = target;
    }
    if (ci == NULL)
        ci = code_slot_find(mi, world, facts, rettype, specsig);
    if (ci == NULL && jl_revalidate_policy(mi, world, facts) != NULL)
        ci = code_slot_find(mi, world, facts, rettype, specsig);
    JL_GC_PUSH1(&ci);
    if (ci == NULL) {
        jl_code_instance_t *ci2 = jl_type_infer_policy(mi, world, SOURCE_MODE_ABI, jl_options.trim, facts);
        if (ci2 != NULL && !jl_is_compiled_codeinst(ci2)) {
            ci = ci2;
            jl_compile_codeinst(ci2);
        }
        ci = code_slot_find(mi, world, facts, rettype, specsig);
    }
    if (ci == NULL || code_slot_force_fallback) {
        ci = code_slot_fallback(mi, rettype, specsig);
        if (ci != NULL)
            CODE_SLOT_COUNT(fallbacks);
    }
    if (ci == NULL) {
        jl_printf(JL_STDERR, "code slot: no code with the caller's result for ");
        jl_static_show(JL_STDERR, (jl_value_t*)mi);
        jl_printf(JL_STDERR, " in world %zu (expected ", world);
        jl_static_show(JL_STDERR, rettype);
        jl_printf(JL_STDERR, ", %s, %s)\n", specsig ? "specsig" : "boxed", facts ? "facts" : "plain");
        if (target != NULL) {
            jl_printf(JL_STDERR, " the target:\n");
            code_slot_show(target);
        }
        jl_printf(JL_STDERR, " the cache:\n");
        for (jl_code_instance_t *c = jl_atomic_load_relaxed(&mi->cache); c; c = jl_atomic_load_relaxed(&c->next))
            code_slot_show(c);
        jl_error("code slot: the callee's code does not match its caller");
    }
    jl_atomic_store_release(slot, ci);
    JL_GC_POP();
    return ci;
}

jl_code_instance_t *jl_compile_method_internal(jl_method_instance_t *mi, size_t world)
{
    return jl_compile_method_internal_policy(mi, world, mi_policy_facts_at(mi, JL_REACH_RUNTIME));
}

// Compiles the code of `mi` of a policy (compiler §2.3): every CodeInstance made here records it.
jl_code_instance_t *jl_compile_method_internal_policy(jl_method_instance_t *mi, size_t world, int facts)
{
    // quick check if we already have a compiled result
    jl_code_instance_t *codeinst = jl_method_compiled_policy(mi, world, facts);
    if (codeinst)
        return codeinst;
    // or a marked one, made valid again (compiler §3.3)
    codeinst = jl_revalidate_policy(mi, world, facts);
    if (codeinst)
        return codeinst;

    // if mi has a better (wider) signature preferred for compilation use that
    // instead and just copy it here for caching
    jl_method_instance_t *mi2 = jl_normalize_to_compilable_mi(mi);
    if (mi2 != mi) {
        jl_code_instance_t *codeinst2 = jl_compile_method_internal_policy(mi2, world, facts);
        jl_code_instance_t *codeinst = jl_get_method_inferred_policy(
                mi, codeinst2->rettype,
                jl_atomic_load_relaxed(&codeinst2->min_world),
                jl_atomic_load_relaxed(&codeinst2->max_world),
                jl_atomic_load_relaxed(&codeinst2->debuginfo),
                jl_atomic_load_relaxed(&codeinst2->edges), facts);
        if (jl_atomic_load_relaxed(&codeinst->invoke) == NULL) {
            codeinst->rettype_const = codeinst2->rettype_const;
            jl_gc_wb(codeinst, codeinst->rettype_const);
            uint8_t specsigflags;
            jl_callptr_t invoke;
            void *fptr;
            jl_read_codeinst_invoke(codeinst2, &specsigflags, &invoke, &fptr, 1);
            if (fptr != NULL) {
                void *prev_fptr = NULL;
                // see jitlayers.cpp for the ordering restrictions here
                if (jl_atomic_cmpswap_acqrel(&codeinst->specptr.fptr, &prev_fptr, fptr)) {
                    jl_atomic_store_release(&codeinst->invoke, invoke);
                    // unspec is probably not specsig, but might be using specptr
                    jl_atomic_fetch_or_relaxed(&codeinst->flags, JL_CI_FLAGS_INVOKE_MATCHES_SPECPTR);
                }
                else {
                    // someone else already compiled it
                    while (!(jl_atomic_load_acquire(&codeinst->flags) & JL_CI_FLAGS_INVOKE_MATCHES_SPECPTR)) {
                        jl_cpu_pause();
                    }
                    // codeinst is now set up fully, safe to return
                }
            }
            else {
                jl_callptr_t prev = NULL;
                jl_atomic_cmpswap_acqrel(&codeinst->invoke, &prev, invoke);
            }
        }
        // don't call record_precompile_statement here, since we already compiled it as mi2 which is better
        return codeinst;
    }

    int compile_option = jl_options.compile_enabled;
    jl_method_t *def = mi->def.method;
    // disabling compilation per-module can override global setting
    if (jl_is_method(def)) {
        int mod_setting = jl_get_module_compile(((jl_method_t*)def)->module);
        if (mod_setting == JL_OPTIONS_COMPILE_OFF ||
            mod_setting == JL_OPTIONS_COMPILE_MIN)
            compile_option = ((jl_method_t*)def)->module->compile;
    }

    // if compilation is disabled or source is unavailable, try calling unspecialized version
    if (compile_option == JL_OPTIONS_COMPILE_OFF ||
        compile_option == JL_OPTIONS_COMPILE_MIN ||
        (jl_is_method(def) && def->source == jl_nothing)) {
        // copy fptr from the template method definition
        if (jl_is_method(def)) {
            jl_method_instance_t *unspecmi = jl_atomic_load_relaxed(&def->unspecialized);
            if (unspecmi) {
                // the unspecialized code of the same policy
                jl_code_instance_t *unspec = jl_atomic_load_relaxed(&unspecmi->cache);
                while (unspec && (unspec->owner != jl_nothing || !ci_policy_is(unspec, facts)))
                    unspec = jl_atomic_load_relaxed(&unspec->next);
                if (unspec && jl_atomic_load_acquire(&unspec->invoke) != NULL) {
                    uint8_t specsigflags;
                    jl_callptr_t invoke;
                    void *fptr;
                    jl_read_codeinst_invoke(unspec, &specsigflags, &invoke, &fptr, 1);
                    jl_debuginfo_t *di = NULL;
                    jl_svec_t *edges = jl_emptysvec;
                    jl_code_instance_t *codeinst = jl_new_codeinst(mi, jl_nothing,
                        (jl_value_t*)jl_any_type, (jl_value_t*)jl_any_type, NULL, NULL,
                        0, 1, ~(size_t)0, 0, jl_nothing, di, edges);
                    jl_ci_set_policy(codeinst, facts);
                    codeinst->rettype_const = unspec->rettype_const;
                    jl_atomic_store_relaxed(&codeinst->specptr.fptr, fptr);
                    jl_atomic_store_relaxed(&codeinst->invoke, invoke);
                    // unspec is probably not specsig, but might be using specptr
                    jl_atomic_store_relaxed(&codeinst->flags, (specsigflags & JL_CI_FLAGS_INVOKE_MATCHES_SPECPTR) |
                        (jl_atomic_load_relaxed(&codeinst->flags) & (JL_CI_FLAGS_FACTS | JL_CI_FLAGS_G2))); // the policy and G2 stay
                    jl_mi_cache_insert(mi, codeinst);
                    record_precompile_statement(mi, 0, 0);
                    return codeinst;
                }
            }
        }
    }

    // if that didn't work and compilation is off, try running in the interpreter
    if (compile_option == JL_OPTIONS_COMPILE_OFF ||
        compile_option == JL_OPTIONS_COMPILE_MIN) {
        jl_code_info_t *src = jl_code_for_interpreter(mi, world);
        if (!jl_code_requires_compiler(src, 0)) {
            jl_debuginfo_t *di = NULL;
            jl_svec_t *edges = jl_emptysvec;
            jl_code_instance_t *codeinst = jl_new_codeinst(mi, jl_nothing,
                (jl_value_t*)jl_any_type, (jl_value_t*)jl_any_type, NULL, NULL,
                0, 1, ~(size_t)0, 0, jl_nothing, di, edges);
            jl_ci_set_policy(codeinst, facts);
            jl_atomic_store_release(&codeinst->invoke, jl_fptr_interpret_call);
            jl_mi_cache_insert(mi, codeinst);
            record_precompile_statement(mi, 0, 0);
            return codeinst;
        }
        if (compile_option == JL_OPTIONS_COMPILE_OFF) {
            jl_printf(JL_STDERR, "No compiled code available for ");
            jl_static_show(JL_STDERR, (jl_value_t*)mi);
            jl_printf(JL_STDERR, " : sysimg may not have been built with --compile=all\n");
        }
    }

    // Ok, compilation is enabled. We'll need to try to compile something (probably).

    // Everything from here on is considered (user facing) compile time
    uint64_t compilation_start = jl_hrtime();
    uint64_t inference_start = jl_typeinf_timing_begin(); // Special-handling for reentrancy

    // Is a recompile if there is cached code, and it was compiled (not only inferred) before
    int is_recompile = 0;
    jl_code_instance_t *codeinst_old = jl_atomic_load_relaxed(&mi->cache);
    while (codeinst_old != NULL) {
        if (jl_atomic_load_relaxed(&codeinst_old->invoke) != NULL) {
            is_recompile = 1;
            break;
        }
        codeinst_old = jl_atomic_load_relaxed(&codeinst_old->next);
    }

    // jl_type_infer will internally do a cache lookup and jl_engine_reserve call
    // to synchronize this across threads
    if (!codeinst) {
        // Don't bother inferring toplevel thunks or macros - the performance cost of inference is likely
        // to significantly exceed the actual runtime.
        int should_skip_inference = !jl_is_method(mi->def.method) || jl_method_is_macro(mi->def.method);

        if (!should_skip_inference) {
            codeinst = jl_type_infer_policy(mi, world, SOURCE_MODE_ABI, jl_options.trim, facts);
        }
    }

    if (codeinst) {
        if (jl_is_compiled_codeinst(codeinst)) {
            jl_typeinf_timing_end(inference_start, is_recompile);
            // Already compiled - e.g. constabi, or compiled by a different thread while we were waiting.
            return codeinst;
        }

        JL_GC_PUSH1(&codeinst);
        int did_compile = jl_compile_codeinst(codeinst);
        double compile_time = jl_hrtime() - compilation_start;

        if (jl_atomic_load_relaxed(&codeinst->invoke) == NULL) {
            // Something went wrong. Bail to the fallback path.
            codeinst = NULL;
        }
        else if (did_compile && codeinst->owner == jl_nothing) {
            record_precompile_statement(mi, compile_time, is_recompile);
        }
        JL_GC_POP();
    }
    if (!codeinst) {
        jl_method_instance_t *unspec = jl_get_unspecialized(def);
        if ((jl_value_t*)unspec == jl_nothing)
            unspec = mi;
        jl_code_instance_t *ucache = jl_get_method_inferred_policy(unspec, (jl_value_t*)jl_any_type, 1, ~(size_t)0, NULL, NULL, facts);
        // ask codegen to make the fptr for unspec
        jl_callptr_t ucache_invoke = jl_atomic_load_acquire(&ucache->invoke);
        if (ucache_invoke == NULL) {
            if ((!jl_is_method(def) || def->source == jl_nothing) &&
                !jl_cached_uninferred(jl_atomic_load_relaxed(&jl_get_ci_mi(ucache)->cache), world)) {
                jl_throw(jl_new_struct(jl_missingcodeerror_type, (jl_value_t*)mi));
            }
            jl_generate_fptr_for_unspecialized(ucache);
            ucache_invoke = jl_atomic_load_acquire(&ucache->invoke);
        }
        assert(ucache_invoke != NULL);
        if (ucache_invoke != jl_fptr_sparam &&
            ucache_invoke != jl_fptr_interpret_call) {
            // only these care about the exact specTypes, otherwise we can use it directly
            jl_typeinf_timing_end(inference_start, is_recompile);
            return ucache;
        }
        uint8_t specsigflags;
        jl_callptr_t invoke;
        void *fptr;
        jl_read_codeinst_invoke(ucache, &specsigflags, &invoke, &fptr, 1);
        jl_debuginfo_t *di = NULL;
        jl_svec_t *edges = jl_emptysvec;
        codeinst = jl_new_codeinst(mi, jl_nothing,
            (jl_value_t*)jl_any_type, (jl_value_t*)jl_any_type, NULL, NULL,
            0, 1, ~(size_t)0, 0, jl_nothing, di, edges);
        jl_ci_set_policy(codeinst, facts);
        codeinst->rettype_const = ucache->rettype_const;
        // unspec is always not specsig, but might use specptr
        jl_atomic_store_relaxed(&codeinst->specptr.fptr, fptr);
        jl_atomic_store_relaxed(&codeinst->invoke, invoke);
        jl_atomic_store_relaxed(&codeinst->flags, (specsigflags & JL_CI_FLAGS_INVOKE_MATCHES_SPECPTR) |
                        (jl_atomic_load_relaxed(&codeinst->flags) & (JL_CI_FLAGS_FACTS | JL_CI_FLAGS_G2))); // the policy and G2 stay
        jl_mi_cache_insert(mi, codeinst);
    }
    jl_atomic_store_relaxed(&codeinst->precompile, 1);
    jl_typeinf_timing_end(inference_start, is_recompile);
    return codeinst;
}

jl_value_t *jl_fptr_const_return(jl_value_t *f, jl_value_t **args, uint32_t nargs, jl_code_instance_t *m)
{
    return m->rettype_const;
}

jl_value_t *jl_fptr_args(jl_value_t *f, jl_value_t **args, uint32_t nargs, jl_code_instance_t *m)
{
    jl_fptr_args_t invoke = jl_atomic_load_relaxed(&m->specptr.fptr1);
    assert(invoke && "Forgot to set specptr for jl_fptr_args!");
    return invoke(f, args, nargs);
}

jl_value_t *jl_fptr_sparam(jl_value_t *f, jl_value_t **args, uint32_t nargs, jl_code_instance_t *m)
{
    jl_svec_t *sparams = jl_get_ci_mi(m)->sparam_vals;
    assert(sparams != jl_emptysvec);
    jl_fptr_sparam_t invoke = jl_atomic_load_relaxed(&m->specptr.fptr3);
    assert(invoke && "Forgot to set specptr for jl_fptr_sparam!");
    return invoke(f, args, nargs, sparams);
}

jl_value_t *jl_fptr_wait_for_compiled(jl_value_t *f, jl_value_t **args, uint32_t nargs, jl_code_instance_t *m)
{
    jl_callptr_t invoke = jl_atomic_load_acquire(&m->invoke);
    if (invoke == &jl_fptr_wait_for_compiled) {
        int64_t last_alloc = jl_options.malloc_log ? jl_gc_diff_total_bytes() : 0;
        int last_errno = errno;
#ifdef _OS_WINDOWS_
        DWORD last_error = GetLastError();
#endif
        jl_compile_codeinst(m);
#ifdef _OS_WINDOWS_
        SetLastError(last_error);
#endif
        errno = last_errno;
        if (jl_options.malloc_log)
            jl_gc_sync_total_bytes(last_alloc); // discard allocation count from compilation
        invoke = jl_atomic_load_acquire(&m->invoke);
    }
    return invoke(f, args, nargs, m);
}

// test whether codeinst->invoke is usable already without further compilation needed
JL_DLLEXPORT int jl_is_compiled_codeinst(jl_code_instance_t *codeinst)
{
    jl_callptr_t invoke = jl_atomic_load_relaxed(&codeinst->invoke);
    if (invoke == NULL || invoke == &jl_fptr_wait_for_compiled)
        return 0;
    return 1;
}

JL_DLLEXPORT const jl_callptr_t jl_fptr_args_addr = &jl_fptr_args;

JL_DLLEXPORT const jl_callptr_t jl_fptr_const_return_addr = &jl_fptr_const_return;

JL_DLLEXPORT const jl_callptr_t jl_fptr_sparam_addr = &jl_fptr_sparam;

JL_CALLABLE(jl_f_opaque_closure_call);
JL_DLLEXPORT const jl_callptr_t jl_f_opaque_closure_call_addr = (jl_callptr_t)&jl_f_opaque_closure_call;

JL_DLLEXPORT const jl_callptr_t jl_fptr_wait_for_compiled_addr = &jl_fptr_wait_for_compiled;

// Return the index of the invoke api, if known
JL_DLLEXPORT int32_t jl_invoke_api(jl_code_instance_t *codeinst)
{
    jl_callptr_t f = jl_atomic_load_relaxed(&codeinst->invoke);
    if (f == NULL)
        return 0;
    if (f == &jl_fptr_args)
        return 1;
    if (f == &jl_fptr_const_return)
        return 2;
    if (f == &jl_fptr_sparam)
        return 3;
    if (f == &jl_fptr_interpret_call)
        return 4;
    return -1;
}

// The compilation signature of a call whose caller's inference chose the specialization `ti`. At a
// trait argument, the attribute is inference's choice, which `ti` records (compiler §6.3): the
// declared type where the code is compiled once, the value's type otherwise.
JL_DLLEXPORT jl_value_t *jl_normalize_to_compilable_sig(jl_tupletype_t *ti, jl_svec_t *env, jl_method_t *m,
                                                        int return_if_compileable)
{
    uint32_t force_mask = jl_is_datatype(ti) ? trait_arg_bits(m) : 0, force_dynamic = 0;
    for (size_t i = 1; force_mask && i <= 32 && i < jl_nparams(ti); i++)
        if ((force_mask & (1u << (i - 1))) && jl_egal(jl_tparam(ti, i), jl_nth_slot_type(m->sig, i)))
            force_dynamic |= 1u << (i - 1);
    jl_tupletype_t *tt = NULL;
    jl_svec_t *newparams = NULL;
    JL_GC_PUSH2(&tt, &newparams);
    intptr_t max_varargs = get_max_varargs(m, NULL);
    jl_compilation_sig(ti, env, m, max_varargs,
                       effective_nospecialize(m, ti, 1, force_mask, force_dynamic), &newparams);
    int is_compileable = ((jl_datatype_t*)ti)->isdispatchtuple;
    if (newparams) {
        tt = (jl_datatype_t*)jl_apply_tuple_type(newparams, 1);
        if (!is_compileable) {
            // compute new env, if used below
            jl_value_t *ti = jl_type_intersection_env((jl_value_t*)tt, (jl_value_t*)m->sig, &newparams);
            assert(ti != jl_bottom_type); (void)ti;
            env = newparams;
        }
    }
    else {
        tt = ti;
    }
    if (!is_compileable)
        is_compileable = jl_isa_compileable_sig(tt, env, m);
    JL_GC_POP();
    return (!return_if_compileable || is_compileable) ? (jl_value_t*)tt : jl_nothing;
}


jl_method_instance_t *jl_normalize_to_compilable_mi(jl_method_instance_t *mi JL_PROPAGATES_ROOT)
{
    jl_method_t *def = mi->def.method;
    if (!jl_is_method(def) || !jl_is_datatype(mi->specTypes) || def->is_for_opaque_closure)
        return mi;
    jl_value_t *compilationsig = jl_normalize_to_compilable_sig((jl_datatype_t*)mi->specTypes, mi->sparam_vals, def, 1);
    if (compilationsig == jl_nothing || jl_egal(compilationsig, mi->specTypes))
        return mi;
    jl_svec_t *env = NULL;
    JL_GC_PUSH2(&compilationsig, &env);
    jl_value_t *ti = jl_type_intersection_env((jl_value_t*)compilationsig, (jl_value_t*)def->sig, &env);
    assert(ti != jl_bottom_type); (void)ti;
    mi = jl_specializations_get_linfo(def, (jl_value_t*)compilationsig, env);
    JL_GC_POP();
    return mi;
}

// return a MethodInstance for a compileable method_match, if valid
static jl_value_t *jl_method_match_to_mi(jl_method_match_t *match, size_t world, size_t min_valid, size_t max_valid, int mt_cache)
{
    jl_method_t *m = match->method;
    JL_GC_PROMISE_ROOTED(m);
    jl_svec_t *env = match->sparams;
    jl_tupletype_t *ti = match->spec_types;
    jl_value_t *mi = jl_nothing;
    if (jl_is_datatype(ti)) {
        // get the specialization, possibly also caching it
        if (mt_cache && ((jl_datatype_t*)ti)->isdispatchtuple) {
            // Since we also use this presence in the cache
            // to trigger compilation when producing `.ji` files,
            // inject it there now if we think it will be
            // used via dispatch later (e.g. because it was hinted via a call to `precompile`)
            jl_methcache_t *mc = jl_method_table->cache;
            assert(mc);
            JL_LOCK(&mc->writelock);
            // this path does not pre-probe `ti`, so cache_method must keep its
            // own presence check (ti may already be cached from a prior hint)
            mi = (jl_value_t*)cache_method(jl_method_get_table(m), mc, &mc->cache, (jl_value_t*)mc, ti, m, world, min_valid, max_valid, env, /*tt_known_absent*/0);
        }
        else {
            jl_value_t *tt = jl_normalize_to_compilable_sig(ti, env, m, 1);
            if (tt != jl_nothing) {
                JL_GC_PUSH2(&tt, &env);
                if (!jl_egal(tt, (jl_value_t*)ti)) {
                    jl_value_t *ti = jl_type_intersection_env((jl_value_t*)tt, (jl_value_t*)m->sig, &env);
                    assert(ti != jl_bottom_type); (void)ti;
                }
                mi = (jl_value_t*)jl_specializations_get_linfo(m, (jl_value_t*)tt, env);
                JL_GC_POP();
            }
        }
    }
    return mi;
}

// compile-time method lookup
// intersect types with the MT, and return a single compileable specialization that covers the intersection.
jl_value_t *jl_get_specialization1(jl_tupletype_t *types, size_t world, int mt_cache)
{
    if (jl_has_free_typevars((jl_value_t*)types))
        return jl_nothing; // don't poison the cache due to a malformed query
    if (!jl_has_concrete_subtype((jl_value_t*)types))
        return jl_nothing;

    // find if exactly 1 method matches (issue #7302)
    size_t min_valid2 = 1;
    size_t max_valid2 = ~(size_t)0;
    int ambig = 0;
    jl_value_t *matches = jl_matching_methods(types, jl_nothing, 1, 1, world, &min_valid2, &max_valid2, &ambig);
    if (matches == jl_nothing || jl_array_nrows(matches) != 1 || ambig)
        return jl_nothing;
    JL_GC_PUSH1(&matches);
    jl_method_match_t *match = (jl_method_match_t*)jl_array_ptr_ref(matches, 0);
    jl_value_t *mi = jl_method_match_to_mi(match, world, min_valid2, max_valid2, mt_cache);
    JL_GC_POP();
    return mi;
}

// jl_get_specialization1, with the range of worlds where its answer holds (for the calling
// convention `:caller_world` of cfunction, runtime_ccall.cpp)
JL_DLLEXPORT jl_value_t *jl_get_specialization1_worlds(jl_tupletype_t *types, size_t world, size_t *min_valid, size_t *max_valid)
{
    if (jl_has_free_typevars((jl_value_t*)types))
        return jl_nothing;
    if (!jl_has_concrete_subtype((jl_value_t*)types))
        return jl_nothing;
    int ambig = 0;
    jl_value_t *matches = jl_matching_methods(types, jl_nothing, 1, 1, world, min_valid, max_valid, &ambig);
    if (matches == jl_nothing || jl_array_nrows(matches) != 1 || ambig)
        return jl_nothing;
    JL_GC_PUSH1(&matches);
    jl_method_match_t *match = (jl_method_match_t*)jl_array_ptr_ref(matches, 0);
    jl_value_t *mi = jl_method_match_to_mi(match, world, *min_valid, *max_valid, 0);
    JL_GC_POP();
    return mi;
}

// Try to get a MethodInstance for a precompile() call. This uses a special kind of lookup that
// tries to find a method for which the requested signature is compileable.
JL_DLLEXPORT jl_value_t *jl_get_compile_hint_specialization(jl_tupletype_t *types JL_PROPAGATES_ROOT, size_t world, int mt_cache)
{
    if (jl_has_free_typevars((jl_value_t*)types))
        return jl_nothing; // don't poison the cache due to a malformed query
    if (!jl_has_concrete_subtype((jl_value_t*)types))
        return jl_nothing;

    size_t min_valid2 = 1;
    size_t max_valid2 = ~(size_t)0;
    int ambig = 0;
    jl_value_t *matches = jl_matching_methods(types, jl_nothing, -1, 0, world, &min_valid2, &max_valid2, &ambig);
    size_t i, n = jl_array_nrows(matches);
    if (n == 0)
        return jl_nothing;
    JL_GC_PUSH1(&matches);
    jl_method_match_t *match = NULL;
    if (n == 1) {
        match = (jl_method_match_t*)jl_array_ptr_ref(matches, 0);
    }
    else if (jl_is_datatype(types)) {
        // first, select methods for which `types` is compileable
        size_t count = 0;
        for (i = 0; i < n; i++) {
            jl_method_match_t *match1 = (jl_method_match_t*)jl_array_ptr_ref(matches, i);
            if (jl_isa_compileable_sig(types, match1->sparams, match1->method))
                jl_array_ptr_set(matches, count++, (jl_value_t*)match1);
        }
        jl_array_del_end((jl_array_t*)matches, n - count);
        n = count;
        // now remove methods that are more specific than others in the list.
        // this is because the intent of precompiling e.g. f(::DataType) is to
        // compile that exact method if it exists, and not lots of f(::Type{X}) methods
        int exclude;
        count = 0;
        for (i = 0; i < n; i++) {
            jl_method_match_t *match1 = (jl_method_match_t*)jl_array_ptr_ref(matches, i);
            exclude = 0;
            for (size_t j = n-1; j > i; j--) {  // more general methods maybe more likely to be at end
                jl_method_match_t *match2 = (jl_method_match_t*)jl_array_ptr_ref(matches, j);
                if (jl_method_morespecific(match1->method, match2->method)) {
                    exclude = 1;
                    break;
                }
            }
            if (!exclude)
                jl_array_ptr_set(matches, count++, (jl_value_t*)match1);
            if (count > 1)
                break;
        }
        // at this point if there are 0 matches left we found nothing, or if there are
        // more than one the request is ambiguous and we ignore it.
        if (count == 1)
            match = (jl_method_match_t*)jl_array_ptr_ref(matches, 0);
    }
    jl_value_t *mi = jl_nothing;
    if (match != NULL)
        mi = jl_method_match_to_mi(match, world, min_valid2, max_valid2, mt_cache);
    JL_GC_POP();
    return mi;
}

static void _generate_from_hint(jl_method_instance_t *mi, size_t world)
{
    jl_value_t *codeinst = jl_rettype_inferred_native(mi, world, world);
    if (codeinst == jl_nothing) {
        (void)jl_type_infer(mi, world, SOURCE_MODE_NOT_REQUIRED, jl_options.trim);
        codeinst = jl_rettype_inferred_native(mi, world, world);
    }
    if (codeinst != jl_nothing) {
        if (jl_atomic_load_relaxed(&((jl_code_instance_t*)codeinst)->invoke) == jl_fptr_const_return)
            return; // probably not a good idea to generate code
        jl_atomic_store_relaxed(&((jl_code_instance_t*)codeinst)->precompile, 1);
    }
}

static void jl_compile_now(jl_method_instance_t *mi)
{
    size_t world = jl_atomic_load_acquire(&jl_world_counter);
    size_t tworld = jl_typeinf_world;
    _generate_from_hint(mi, world);
    if (jl_typeinf_func && jl_atomic_load_relaxed(&mi->def.method->primary_world) <= tworld) {
        // if it's part of the compiler, also attempt to compile for the compiler world too
        _generate_from_hint(mi, tworld);
    }
}

JL_DLLEXPORT void jl_compile_method_instance(jl_method_instance_t *mi, jl_tupletype_t *types, size_t world)
{
    size_t tworld = jl_typeinf_world;
    uint8_t miflags = jl_atomic_load_relaxed(&mi->flags) | JL_MI_FLAGS_MASK_PRECOMPILED;
    jl_atomic_store_relaxed(&mi->flags, miflags);
    if (jl_generating_output()) {
        jl_compile_now(mi);
        // In addition to full compilation of the compilation-signature, if `types` is more specific (e.g. due to nospecialize),
        // also run inference now on the original `types`, since that may help us guide inference to find
        // additional useful methods that should be compiled
        //ALT: if (jl_is_datatype(types) && ((jl_datatype_t*)types)->isdispatchtuple && !jl_egal(mi->specTypes, types))
        //ALT: if (jl_subtype(types, mi->specTypes))
        if (types && !jl_subtype(mi->specTypes, (jl_value_t*)types)) {
            jl_svec_t *tpenv2 = jl_emptysvec;
            jl_value_t *types2 = NULL;
            JL_GC_PUSH2(&tpenv2, &types2);
            types2 = jl_type_intersection_env((jl_value_t*)types, (jl_value_t*)mi->def.method->sig, &tpenv2);
            jl_method_instance_t *mi2 = jl_specializations_get_linfo(mi->def.method, (jl_value_t*)types2, tpenv2);
            JL_GC_POP();
            miflags = jl_atomic_load_relaxed(&mi2->flags) | JL_MI_FLAGS_MASK_PRECOMPILED;
            jl_atomic_store_relaxed(&mi2->flags, miflags);
            if (jl_rettype_inferred_native(mi2, world, world) == jl_nothing)
                (void)jl_type_infer(mi2, world, SOURCE_MODE_NOT_REQUIRED, jl_options.trim);
            if (jl_typeinf_func && jl_atomic_load_relaxed(&mi->def.method->primary_world) <= tworld) {
                if (jl_rettype_inferred_native(mi2, tworld, tworld) == jl_nothing)
                    (void)jl_type_infer(mi2, tworld, SOURCE_MODE_NOT_REQUIRED, jl_options.trim);
            }
        }
    }
    else {
        // Otherwise (this branch), assuming we are at runtime (normal JIT) and
        // we should generate the native code immediately in preparation for use.
        (void)jl_compile_method_internal(mi, world);
    }
}

JL_DLLEXPORT void jl_compile_method_sig(jl_method_t *m, jl_value_t *types, jl_svec_t *env, size_t world)
{
    jl_method_instance_t *mi = jl_specializations_get_linfo(m, types, env);
    jl_compile_method_instance(mi, NULL, world);
}

JL_DLLEXPORT int jl_is_compilable(jl_tupletype_t *types)
{
    size_t world = jl_atomic_load_acquire(&jl_world_counter);
    jl_value_t *mi = jl_get_compile_hint_specialization(types, world, 1);
    return mi == jl_nothing ? 0 : 1;
}

JL_DLLEXPORT int jl_compile_hint(jl_tupletype_t *types)
{
    size_t world = jl_atomic_load_acquire(&jl_world_counter);
    jl_value_t *mi = jl_get_compile_hint_specialization(types, world, 1);
    if (mi == jl_nothing)
        return 0;
    JL_GC_PROMISE_ROOTED(mi);
    jl_compile_method_instance((jl_method_instance_t*)mi, types, world);
    return 1;
}


// add type of `f` to front of argument tuple type
jl_value_t *jl_argtype_with_function(jl_value_t *f, jl_value_t *types0)
{
    return jl_argtype_with_function_type(jl_is_type(f) ? (jl_value_t*)jl_wrap_Type(f) : jl_typeof(f), types0);
}

jl_value_t *jl_argtype_with_function_type(jl_value_t *ft JL_MAYBE_UNROOTED, jl_value_t *types0)
{
    jl_value_t *types = jl_unwrap_unionall(types0);
    size_t l = jl_nparams(types);
    jl_value_t *tt = NULL;
    JL_GC_PUSH2(&tt, &ft);
    tt = (jl_value_t*)jl_alloc_svec(1+l);
    jl_svecset(tt, 0, ft);
    for (size_t i = 0; i < l; i++)
        jl_svecset(tt, i+1, jl_tparam(types,i));
    tt = (jl_value_t*)jl_apply_tuple_type((jl_svec_t*)tt, 1);
    tt = jl_rewrap_unionall_(tt, types0);
    JL_GC_POP();
    return tt;
}

// undo jl_argtype_with_function transform
jl_value_t *jl_argtype_without_function(jl_value_t *ftypes)
{
    jl_value_t *types = jl_unwrap_unionall(ftypes);
    size_t l = jl_nparams(types);
    if (l == 1 && jl_is_vararg(jl_tparam0(types)))
        return ftypes;
    jl_value_t *tt = (jl_value_t*)jl_alloc_svec(l - 1);
    JL_GC_PUSH1(&tt);
    for (size_t i = 1; i < l; i++)
        jl_svecset(tt, i - 1, jl_tparam(types, i));
    tt = (jl_value_t*)jl_apply_tuple_type((jl_svec_t*)tt, 0);
    tt = jl_rewrap_unionall_(tt, types);
    JL_GC_POP();
    return tt;
}

#ifdef JL_TRACE
static int trace_en = 0;
static int error_en = 1;
static void __attribute__ ((unused)) enable_trace(int x) { trace_en=x; }
static void show_call(jl_value_t *F, jl_value_t **args, uint32_t nargs)
{
    jl_static_show(JL_STDOUT, F);
    jl_printf(JL_STDOUT, "(");
    for(size_t i=0; i < nargs; i++) {
        if (i > 0) jl_printf(JL_STDOUT, ", ");
        jl_static_show(JL_STDOUT, jl_typeof(args[i]));
    }
    jl_printf(JL_STDOUT, ")");
}
#endif

STATIC_INLINE jl_value_t *verify_type(jl_value_t *v) JL_NOTSAFEPOINT
{
    assert(v && jl_typeof(v) && jl_typeof(jl_typeof(v)) == (jl_value_t*)jl_datatype_type);
    return v;
}

// Runs the code of `mfunc` of the policy it has where `reach` reaches it (compiler §2.3, §2.4).
STATIC_INLINE jl_value_t *_jl_invoke(jl_value_t *F, jl_value_t **args, uint32_t nargs, jl_method_instance_t *mfunc, size_t world, int reach)
{
    jl_code_instance_t *codeinst = NULL;
    int facts = mi_policy_facts_at(mfunc, reach);
    jl_callptr_t invoke = jl_method_compiled_callptr(mfunc, world, facts, &codeinst);
    if (invoke) {
        jl_value_t *res = invoke(F, args, nargs, codeinst);
        return verify_type(res);
    }
    int64_t last_alloc = jl_options.malloc_log ? jl_gc_diff_total_bytes() : 0;
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    codeinst = jl_compile_method_internal_policy(mfunc, world, facts);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    if (jl_options.malloc_log)
        jl_gc_sync_total_bytes(last_alloc); // discard allocation count from compilation
    invoke = jl_atomic_load_acquire(&codeinst->invoke);
    jl_value_t *res = invoke(F, args, nargs, codeinst);
    return verify_type(res);
}

JL_DLLEXPORT jl_value_t *jl_invoke(jl_value_t *F, jl_value_t **args, uint32_t nargs, jl_method_instance_t *mfunc)
{
    size_t world = jl_current_task->world_age;
    return _jl_invoke(F, args, nargs, mfunc, world, JL_REACH_RUNTIME);
}
// the entries of compiled code of each policy, and of a call made for code of a policy (§2.4)
JL_DLLEXPORT jl_value_t *jl_invoke_plain(jl_value_t *F, jl_value_t **args, uint32_t nargs, jl_method_instance_t *mfunc)
{
    size_t world = jl_current_task->world_age;
    return _jl_invoke(F, args, nargs, mfunc, world, JL_REACH_PLAIN);
}
JL_DLLEXPORT jl_value_t *jl_invoke_facts(jl_value_t *F, jl_value_t **args, uint32_t nargs, jl_method_instance_t *mfunc)
{
    size_t world = jl_current_task->world_age;
    return _jl_invoke(F, args, nargs, mfunc, world, JL_REACH_FACTS);
}
JL_DLLEXPORT jl_value_t *jl_invoke_reach(jl_value_t *F, jl_value_t **args, uint32_t nargs, jl_method_instance_t *mfunc, int reach)
{
    size_t world = jl_current_task->world_age;
    return _jl_invoke(F, args, nargs, mfunc, world, reach);
}

JL_DLLEXPORT jl_value_t *jl_invoke_oc(jl_value_t *F, jl_value_t **args, uint32_t nargs, jl_method_instance_t *mfunc)
{
    jl_opaque_closure_t *oc = (jl_opaque_closure_t*)F;
    jl_task_t *ct = jl_current_task;
    size_t last_age = ct->world_age;
    size_t world = oc->world;
    ct->world_age = world;
    jl_value_t *ret = _jl_invoke(F, args, nargs, mfunc, world, JL_REACH_RUNTIME);
    ct->world_age = last_age;
    return ret;
}

// A method-cache entry whose signature has only concrete types and `Any` (no varargs, no type
// variables, no `Type{}` or kinds), no guard signatures, and a simple signature implied by it holds for every type at its `Any`
// positions: Julia gives an entry guard signatures where a more specific method intersects it. The
// per-call-site cache can hold it as it is, matched with `Any` as a wildcard (`sig_match_any`):
// `setindex!(::Array{Any}, @nospecialize(x), ::Int)` at a site that sees 16 value types took 163 ns,
// thrashing the site's four exact entries.
static int call_cache_any_ok(jl_typemap_entry_t *entry) JL_NOTSAFEPOINT
{
    if (entry->isleafsig || entry->va || !entry->issimplesig || entry->guardsigs != jl_emptysvec)
        return 0;
    jl_svec_t *ps = entry->sig->parameters;
    if (entry->simplesig != (void*)jl_nothing) {
        // the method cache matches the simple signature too: it must be implied by the signature
        // (each of its parameters `Any` or the signature's own), so that matching the signature is enough
        jl_svec_t *ss = ((jl_tupletype_t*)entry->simplesig)->parameters;
        if (jl_svec_len(ss) != jl_svec_len(ps))
            return 0;
        for (size_t i = 0; i < jl_svec_len(ss); i++) {
            jl_value_t *q = jl_svecref(ss, i);
            if (q != (jl_value_t*)jl_any_type && q != jl_svecref(ps, i))
                return 0;
        }
    }
    int anyany = 0;
    for (size_t i = 0; i < jl_svec_len(ps); i++) {
        jl_value_t *p = jl_svecref(ps, i);
        if (p == (jl_value_t*)jl_any_type) {
            if (i == 0)
                return 0;                       // the function itself is concrete
            anyany = 1;
            continue;
        }
        if (!jl_is_concrete_type(p) || jl_is_kind(p) || jl_is_type_type(p))
            return 0;
    }
    return anyany;
}
STATIC_INLINE int sig_match_any(jl_value_t *arg1t, jl_value_t **args, jl_value_t **sig, size_t n) JL_NOTSAFEPOINT
{
    if (arg1t != sig[0])
        return 0;
    for (size_t i = 1; i < n; i++) {
        jl_value_t *decl = sig[i];
        if (decl != (jl_value_t*)jl_any_type && jl_typeof(args[i - 1]) != decl)
            return 0;
    }
    return 1;
}

STATIC_INLINE int sig_match_fast(jl_value_t *arg1t, jl_value_t **args, jl_value_t **sig, size_t n)
{
    // NOTE: This function is a huge performance hot spot!!
    if (arg1t != sig[0])
        return 0;
    size_t i;
    for (i = 1; i < n; i++) {
        jl_value_t *decl = sig[i];
        jl_value_t *a = args[i - 1];
        if (jl_typeof(a) != decl) {
            /*
              we are only matching concrete types here, and those types are
              hash-consed, so pointer comparison should work.
            */
            return 0;
        }
    }
    return 1;
}

_Atomic(jl_typemap_entry_t*) call_cache[N_CALL_CACHE] JL_GLOBALLY_ROOTED;
static _Atomic(uint8_t) pick_which[N_CALL_CACHE];

// ---- a dispatch profiler (JULIA_DISPATCH_PROFILE=1) ----
// Counts dynamic calls per callee method and per the path their lookup took (the per-call-site
// cache, the leaf cache, the method cache's search, the full lookup), and per call site (the
// calling code by its return address). Printed at exit, the most-called first;
// `ccall(:jl_dispatch_profile_reset, Cvoid, ())` zeroes the counts, to profile one region of a run.
// A tool for finding hot dynamic calls, off by default: one load and a branch per call.
#define DP_SLOTS 4096
#define DP_SITES 16384
enum { DP_SITE = 0, DP_LEAF = 1, DP_ASSOC = 2, DP_FULL = 3 };
typedef struct { _Atomic(jl_method_t*) m; _Atomic(uint64_t) n[4]; } dp_slot_t;
typedef struct { _Atomic(void*) ip; _Atomic(jl_method_t*) m; _Atomic(uint64_t) n; } dp_site_t;
static dp_slot_t dp_table[DP_SLOTS];
static dp_site_t dp_sites[DP_SITES];
static int dp_on = -1;
static _Thread_local void *dp_ip = NULL;          // the dynamic call's return address (jl_apply_generic)
JL_DLLEXPORT void jl_dispatch_profile_print(void);
static int dp_enabled(void) JL_NOTSAFEPOINT
{
    if (dp_on < 0) {
        const char *e = getenv("JULIA_DISPATCH_PROFILE");
        int on = e && e[0] == '1';
        static _Atomic(int) dp_registered = 0;
        int expected = 0;
        if (on && jl_atomic_cmpswap(&dp_registered, &expected, 1))
            atexit(jl_dispatch_profile_print);
        dp_on = on;
    }
    return dp_on;
}
static void dp_count_site(void *ip, jl_method_t *m) JL_NOTSAFEPOINT
{
    uintptr_t h = (((uintptr_t)ip >> 2) ^ ((uintptr_t)m >> 4)) * 0x9E3779B97F4A7C15ull;
    for (size_t k = 0; k < 64; k++) {
        dp_site_t *s = &dp_sites[(h + k) & (DP_SITES - 1)];
        void *cur = jl_atomic_load_relaxed(&s->ip);
        if (cur == NULL) {
            void *expected = NULL;
            if (!jl_atomic_cmpswap(&s->ip, &expected, ip) && expected != ip)
                continue;
            jl_atomic_store_relaxed(&s->m, m);
            cur = ip;
        }
        if (cur == ip && jl_atomic_load_relaxed(&s->m) == m) {
            jl_atomic_fetch_add_relaxed(&s->n, 1);
            return;
        }
    }
}
static void dp_count(jl_method_instance_t *mfunc, int path) JL_NOTSAFEPOINT
{
    jl_method_t *m = jl_is_method(mfunc->def.method) ? mfunc->def.method : NULL;
    if (m == NULL)
        return;
    uintptr_t h = ((uintptr_t)m >> 4) * 0x9E3779B97F4A7C15ull;
    for (size_t k = 0; k < DP_SLOTS; k++) {
        dp_slot_t *s = &dp_table[(h + k) & (DP_SLOTS - 1)];
        jl_method_t *cur = jl_atomic_load_relaxed(&s->m);
        if (cur == NULL) {
            jl_method_t *expected = NULL;
            if (!jl_atomic_cmpswap(&s->m, &expected, m) && expected != m)
                continue;
            cur = m;
        }
        if (cur == m) {
            jl_atomic_fetch_add_relaxed(&s->n[path], 1);
            if (dp_ip != NULL)
                dp_count_site(dp_ip, m);
            return;
        }
    }
}
JL_DLLEXPORT void jl_dispatch_profile_reset(void)
{
    for (size_t i = 0; i < DP_SLOTS; i++)
        for (int p = 0; p < 4; p++)
            jl_atomic_store_relaxed(&dp_table[i].n[p], 0);
    for (size_t i = 0; i < DP_SITES; i++)
        jl_atomic_store_relaxed(&dp_sites[i].n, 0);
}
JL_DLLEXPORT void jl_dispatch_profile_print(void)
{
    static size_t idx[DP_SLOTS], sidx[DP_SITES];
    static uint64_t tot[DP_SLOTS];
    size_t cnt = 0, scnt = 0;
    for (size_t i = 0; i < DP_SLOTS; i++) {
        if (jl_atomic_load_relaxed(&dp_table[i].m) == NULL)
            continue;
        uint64_t t = 0;
        for (int p = 0; p < 4; p++)
            t += jl_atomic_load_relaxed(&dp_table[i].n[p]);
        tot[i] = t;
        idx[cnt++] = i;
    }
    for (size_t a = 0; a < cnt && a < 40; a++)  // a selection of the top 40 by total
        for (size_t b = a + 1; b < cnt; b++)
            if (tot[idx[b]] > tot[idx[a]]) { size_t x = idx[a]; idx[a] = idx[b]; idx[b] = x; }
    jl_safe_printf("\n-- dynamic calls by callee method (lookup path: site/leaf/assoc/full)\n");
    for (size_t a = 0; a < cnt && a < 40; a++) {
        dp_slot_t *s = &dp_table[idx[a]];
        jl_method_t *m = jl_atomic_load_relaxed(&s->m);
        jl_safe_printf("%12llu  %s.%s  %s:%d   %llu/%llu/%llu/%llu\n",
            (unsigned long long)tot[idx[a]], jl_symbol_name(m->module->name), jl_symbol_name(m->name),
            jl_symbol_name(m->file), (int)m->line,
            (unsigned long long)jl_atomic_load_relaxed(&s->n[0]), (unsigned long long)jl_atomic_load_relaxed(&s->n[1]),
            (unsigned long long)jl_atomic_load_relaxed(&s->n[2]), (unsigned long long)jl_atomic_load_relaxed(&s->n[3]));
    }
    // the call sites: where in the code the dynamic calls are made (innermost frame, with inlining)
    for (size_t i = 0; i < DP_SITES; i++)
        if (jl_atomic_load_relaxed(&dp_sites[i].ip) != NULL)
            sidx[scnt++] = i;
    for (size_t a = 0; a < scnt && a < 40; a++)
        for (size_t b = a + 1; b < scnt; b++)
            if (jl_atomic_load_relaxed(&dp_sites[sidx[b]].n) > jl_atomic_load_relaxed(&dp_sites[sidx[a]].n)) {
                size_t x = sidx[a]; sidx[a] = sidx[b]; sidx[b] = x;
            }
    jl_safe_printf("\n-- dynamic calls by call site (the calling code, innermost frame first; the callee)\n");
    for (size_t a = 0; a < scnt && a < 40; a++) {
        dp_site_t *s = &dp_sites[sidx[a]];
        jl_method_t *m = jl_atomic_load_relaxed(&s->m);
        jl_frame_t *frames = NULL;
        int nf = jl_getFunctionInfo(&frames, (uintptr_t)jl_atomic_load_relaxed(&s->ip) - 1, 0, 0);
        jl_safe_printf("%12llu  ", (unsigned long long)jl_atomic_load_relaxed(&s->n));
        if (nf > 0) {
            jl_frame_t *f = &frames[nf - 1];
            jl_safe_printf("%s at %s:%d", f->func_name ? f->func_name : "?", f->file_name ? f->file_name : "?", f->line);
            if (nf > 1)
                jl_safe_printf(" (in %s at %s:%d)", frames[0].func_name ? frames[0].func_name : "?",
                               frames[0].file_name ? frames[0].file_name : "?", frames[0].line);
        }
        else {
            jl_safe_printf("%p", jl_atomic_load_relaxed(&s->ip));
        }
        jl_safe_printf("  ->  %s.%s\n", m ? jl_symbol_name(m->module->name) : "?", m ? jl_symbol_name(m->name) : "?");
        for (int k = 0; k < nf; k++) {
            free(frames[k].func_name);
            free(frames[k].file_name);
        }
        free(frames);
    }
}

#ifdef JL_GF_PROFILE
size_t ncalls;
void call_cache_stats()
{
    int pick_which_stat[4] = {0, 0, 0, 0};
    int i, count = 0;
    for (i = 0; i < N_CALL_CACHE; i++) {
        if (jl_atomic_load_relaxed(&call_cache[i]))
            count++;
        ++pick_which_stat[jl_atomic_load_relaxed(&pick_which[i]) & 3];
    }
    jl_safe_printf("cache occupied: %d / %d; pick_which stats: {%d, %d, %d, %d}\n",
            count, N_CALL_CACHE,
            pick_which_stat[0], pick_which_stat[1], pick_which_stat[2], pick_which_stat[3]);
}
#endif

// The member hook (compiler §10.4): hook(T, X) checks the member T of a pattern's membership in
// the set X at its first use. The library sets it; it is rooted for the collectors.
jl_value_t *jl_member_hook JL_GLOBALLY_ROOTED = NULL;
JL_DLLEXPORT void jl_set_member_hook(jl_value_t *hook)
{
    jl_member_hook = hook == jl_nothing ? NULL : hook;
}

// Fully typed mode (compiler §2.7): the compiler puts a call of `jl_not_fully_typed` at the start of
// code that fails it, with the site `svec(mi, level, warned, kind, rt)`. The library's hook (site)
// warns or raises; under warn it runs once per site. Run in the latest world.
jl_value_t *jl_fully_typed_hook JL_GLOBALLY_ROOTED = NULL;
JL_DLLEXPORT void jl_set_fully_typed_hook(jl_value_t *hook)
{
    jl_fully_typed_hook = hook == jl_nothing ? NULL : hook;
}
JL_DLLEXPORT void jl_not_fully_typed(jl_value_t *site)
{
    if (!jl_is_svec(site) || jl_svec_len(site) < 3 || jl_svecref(site, 2) != jl_false)
        return;
    int enforce = jl_unbox_long(jl_svecref(site, 1)) == 2;
    if (!enforce)
        jl_svecset(site, 2, jl_true);
    jl_value_t *hook = jl_fully_typed_hook;
    if (hook == NULL) {
        if (enforce)
            jl_error("the code is not fully typed (fully typed mode)");
        return;
    }
    jl_task_t *ct = jl_current_task;
    size_t last_age = ct->world_age;
    ct->world_age = jl_atomic_load_acquire(&jl_world_counter);
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 2);
    args[0] = hook;
    args[1] = site;
    JL_TRY {
        jl_apply(args, 2);
    }
    JL_CATCH {
        ct->world_age = last_age;
        jl_rethrow();
    }
    ct->world_age = last_age;
    JL_GC_POP();
}

// The library's definition, image and module-end hooks (compiler §10.1). Each runs in the latest
// world, outside the world counter's lock, not for a definition a hook makes, and not while the
// system image is built. They are rooted for the collectors.
jl_value_t *jl_method_definition_hook JL_GLOBALLY_ROOTED = NULL;
jl_value_t *jl_image_methods_hook JL_GLOBALLY_ROOTED = NULL;
jl_value_t *jl_module_end_hook JL_GLOBALLY_ROOTED = NULL;
JL_DLLEXPORT void jl_set_method_definition_hook(jl_value_t *hook)
{
    jl_method_definition_hook = hook == jl_nothing ? NULL : hook;
}
JL_DLLEXPORT void jl_set_image_methods_hook(jl_value_t *hook)
{
    jl_image_methods_hook = hook == jl_nothing ? NULL : hook;
}
JL_DLLEXPORT void jl_set_module_end_hook(jl_value_t *hook)
{
    jl_module_end_hook = hook == jl_nothing ? NULL : hook;
}
// Strict mode's deletion hook (compiler §10.2): hook(m), before a method is deleted; it may throw to
// refuse the deletion.
jl_value_t *jl_method_deletion_hook JL_GLOBALLY_ROOTED = NULL;
JL_DLLEXPORT void jl_set_method_deletion_hook(jl_value_t *hook)
{
    jl_method_deletion_hook = hook == jl_nothing ? NULL : hook;
}

static _Thread_local int library_hook_depth = 0;

// args[0] is the hook; returns its result, or NULL where it does not run
static jl_value_t *call_library_hook(jl_value_t **args, int nargs)
{
    if (args[0] == NULL || library_hook_depth > 0 || (jl_generating_output() && !jl_options.incremental))
        return NULL;
    jl_task_t *ct = jl_current_task;
    size_t last_age = ct->world_age;
    ct->world_age = jl_atomic_load_acquire(&jl_world_counter);
    library_hook_depth++;
    jl_value_t *r = NULL;
    JL_TRY {
        r = jl_apply(args, nargs);
    }
    JL_CATCH {
        library_hook_depth--;
        ct->world_age = last_age;
        jl_rethrow();
    }
    library_hook_depth--;
    ct->world_age = last_age;
    return r;
}

// before a method enters its table (it may throw to refuse it), and once it is visible
void jl_call_definition_hook(jl_method_t *m, int after)
{
    if (jl_method_definition_hook == NULL)
        return;
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 3);
    args[0] = jl_method_definition_hook;
    args[1] = (jl_value_t*)m;
    args[2] = after ? jl_true : jl_false;
    call_library_hook(args, 3);
    JL_GC_POP();
}

// before a method is deleted (it may throw to refuse it)
void jl_call_deletion_hook(jl_method_t *m)
{
    if (jl_method_deletion_hook == NULL)
        return;
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 2);
    args[0] = jl_method_deletion_hook;
    args[1] = (jl_value_t*)m;
    call_library_hook(args, 2);
    JL_GC_POP();
}

// once per package image, with its methods for other modules' functions, before any is activated:
// the methods the hook returns stay inactive. `entries` holds the image's typemap entries for
// them (jl_add_methods made them); returns the entries to activate.
static jl_value_t *entry_method(jl_value_t *e) JL_NOTSAFEPOINT
{
    return jl_typetagis(e, jl_typemap_entry_type) ? (jl_value_t*)((jl_typemap_entry_t*)e)->func.method : e;
}
jl_array_t *jl_call_image_methods_hook(jl_array_t *entries, const char *pkgname)
{
    if (jl_image_methods_hook == NULL || jl_array_nrows(entries) == 0)
        return entries;
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 5);
    jl_array_t *ms = jl_alloc_vec_any(0);
    args[4] = (jl_value_t*)ms;
    for (size_t i = 0; i < jl_array_nrows(entries); i++)
        jl_array_ptr_1d_push(ms, entry_method(jl_array_ptr_ref(entries, i)));
    args[0] = jl_image_methods_hook;
    args[1] = (jl_value_t*)ms;
    args[2] = jl_cstr_to_string(pkgname ? pkgname : "");
    args[3] = call_library_hook(args, 3);
    jl_array_t *keep = entries;
    if (args[3] != NULL && jl_is_array(args[3]) && jl_array_nrows((jl_array_t*)args[3]) > 0) {
        jl_array_t *drop = (jl_array_t*)args[3];
        keep = jl_alloc_vec_any(0);
        args[1] = (jl_value_t*)keep;
        for (size_t i = 0; i < jl_array_nrows(entries); i++) {
            jl_value_t *e = jl_array_ptr_ref(entries, i);
            jl_value_t *m = entry_method(e);
            int dropped = 0;
            for (size_t j = 0; j < jl_array_nrows(drop) && !dropped; j++)
                dropped = jl_array_ptr_ref(drop, j) == m;
            if (!dropped)
                jl_array_ptr_1d_push(keep, e);
        }
    }
    JL_GC_POP();
    return keep;
}

// when a module's definition ends, before its __init__ runs
void jl_call_module_end_hook(jl_module_t *m)
{
    if (jl_module_end_hook == NULL)
        return;
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 2);
    args[0] = jl_module_end_hook;
    args[1] = (jl_value_t*)m;
    call_library_hook(args, 2);
    JL_GC_POP();
}

// whether a type or a type on its supertype chain holds membership entries
static int chain_has_memberships(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (!jl_is_datatype(t))
        return 0;
    for (jl_datatype_t *dt = (jl_datatype_t*)t; dt != jl_any_type; dt = dt->super)
        if (jl_svec_len(jl_typename_memberships(dt->name)) != 0)
            return 1;
    return 0;
}

// After a failed check: lower the method-cache entries for these argument types that the check
// itself may have made (its method matching caches), so the next call misses and asks the hook
// again, which raises the kept failure.
static void member_hook_uncache(jl_tupletype_t *tt)
{
    jl_methcache_t *mc = jl_method_table->cache;
    size_t world = jl_atomic_load_acquire(&jl_world_counter);
    JL_LOCK(&mc->writelock);
    for (;;) {
        jl_typemap_entry_t *entry = jl_mt_find_cache_entry(mc, (jl_datatype_t*)tt, world);
        if (entry == NULL)
            break;
        jl_atomic_store_relaxed(&entry->max_world, world - 1);
    }
    JL_UNLOCK(&mc->writelock);
}

// C-71 (compiler §10.4): after a failed check, also the cache entries the check itself made for other
// calls at the member (its method matching for the set's other operations): every live entry whose
// signature names the member at an argument, and the leaf cache, dropped whole as an insertion does.
// A failed check is an error path, so the walk's cost does not matter.
struct uncache_member_env {
    jl_value_t *member;
    size_t max_world;
};
static int uncache_member_visitor(jl_typemap_entry_t *e, void *closure0)
{
    struct uncache_member_env *env = (struct uncache_member_env*)closure0;
    if (jl_atomic_load_relaxed(&e->max_world) != ~(size_t)0)
        return 1;
    jl_value_t *sig = jl_unwrap_unionall((jl_value_t*)e->sig);
    if (jl_is_datatype(sig)) {
        for (size_t i = 1; i < jl_nparams(sig); i++) {
            if (jl_tparam(sig, i) == env->member) {
                jl_atomic_store_relaxed(&e->max_world, env->max_world);
                break;
            }
        }
    }
    return 1;
}
static void member_hook_uncache_member(jl_value_t *member)
{
    jl_methcache_t *mc = jl_method_table->cache;
    size_t world = jl_atomic_load_acquire(&jl_world_counter);
    struct uncache_member_env env = {member, world - 1};
    JL_LOCK(&mc->writelock);
    jl_typemap_visitor(jl_atomic_load_relaxed(&mc->cache), uncache_member_visitor, (void*)&env);
    jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
    for (size_t li = 1; li < leafcache->length; li += 2) {
        jl_value_t *entry = jl_genericmemory_ptr_ref(leafcache, li);
        while (entry != NULL && entry != jl_nothing) {
            jl_atomic_store_relaxed(&((jl_typemap_entry_t*)entry)->max_world, world - 1);
            entry = (jl_value_t*)jl_atomic_load_relaxed(&((jl_typemap_entry_t*)entry)->next);
        }
    }
    jl_atomic_store_relaxed(&mc->leafcache, (jl_genericmemory_t*)jl_an_empty_memory_any);
    JL_UNLOCK(&mc->writelock);
}

// At dispatch, before the instance lookup of a call that a method at a set serves: for each
// argument at a position the method types by a set, and a member through a membership, the
// library checks the member at its first use. The hook runs in the latest world with no lock
// held, and may throw: then nothing is cached, and the next call asks it again (the library
// keeps its answer, a failure too).
static void member_hook_at_dispatch(jl_tupletype_t *tt, size_t world)
{
    size_t n = jl_nparams(tt);
    int any = 0;
    for (size_t i = 1; i < n && !any; i++)
        any = chain_has_memberships(jl_tparam(tt, i));
    if (!any)
        return;
    size_t min_valid = 0, max_valid = ~(size_t)0;
    jl_method_match_t *matc = _gf_invoke_lookup((jl_value_t*)tt, jl_method_table, world, 0, &min_valid, &max_valid);
    if (matc == NULL)
        return;
    jl_value_t **roots;
    JL_GC_PUSHARGS(roots, 4);
    roots[0] = (jl_value_t*)matc;
    jl_value_t *sig = jl_unwrap_unionall((jl_value_t*)matc->method->sig);
    size_t ns = jl_nparams(sig);
    if (jl_has_open_set_in(sig) && ns > 0) {
        for (size_t i = 1; i < n; i++) {
            size_t j = i < ns ? i : ns - 1;
            jl_value_t *p = jl_tparam(sig, j);
            if (jl_is_vararg(p))
                p = jl_unwrap_vararg(p);
            else if (i >= ns)
                break;
            if (p == NULL)
                continue;
            if (jl_is_typevar(p))
                p = ((jl_tvar_t*)p)->ub;
            p = jl_unwrap_unionall(p);
            if (!jl_is_datatype(p) || !((jl_datatype_t*)p)->name->open_set)
                continue;
            jl_value_t *a = jl_tparam(tt, i);
            if (!chain_has_memberships(a))
                continue; // a declared subtype, or no member at all
            roots[1] = jl_member_hook;
            roots[2] = a;
            roots[3] = ((jl_datatype_t*)p)->name->wrapper;
            jl_task_t *ct = jl_current_task;
            size_t last_age = ct->world_age;
            ct->world_age = jl_atomic_load_acquire(&jl_world_counter);
            JL_TRY {
                jl_apply(&roots[1], 3);
            }
            JL_CATCH {
                member_hook_uncache(tt);
                member_hook_uncache_member(roots[2]);
                jl_rethrow();
            }
            ct->world_age = last_age;
        }
    }
    JL_GC_POP();
}

STATIC_INLINE jl_method_instance_t *jl_lookup_generic_(jl_value_t *F, jl_value_t **args, uint32_t nargs,
                                                       uint32_t callsite, size_t world)
{
#ifdef JL_GF_PROFILE
    ncalls++;
#endif
#ifdef JL_TRACE
    int traceen = trace_en; //&& ((char*)&mt < jl_stack_hi-6000000);
    if (traceen)
        show_call(F, args, nargs);
#endif
    nargs++; // add F to argument count
    jl_value_t *FT = jl_typeof(F);

    /*
      search order:
      check associative hash based on callsite address for leafsig match
      look at concrete signatures
      if there is an exact match, return it
      otherwise look for a matching generic signature
      if no concrete or generic match, raise error
      if no generic match, use the concrete one even if inexact
      otherwise instantiate the generic method and use it
    */
    // compute the entry hashes
    // use different parts of the value
    // so that a collision across all of
    // them is less likely
    uint32_t cache_idx[4] = {
        (callsite) & (N_CALL_CACHE - 1),
        (callsite >> 8) & (N_CALL_CACHE - 1),
        (callsite >> 16) & (N_CALL_CACHE - 1),
        (callsite >> 24 | callsite << 8) & (N_CALL_CACHE - 1)};
    jl_typemap_entry_t *entry = NULL;
    int i;
    jl_tupletype_t *tt = NULL;
    int64_t last_alloc = 0;
    int dp_path = DP_SITE;                          // the dispatch profiler's path (above)
    // check each cache entry to see if it matches
    //#pragma unroll
    //for (i = 0; i < 4; i++) {
    //    LOOP_BODY(i);
    //}
#define LOOP_BODY(_i) do { \
            i = _i; \
            entry = jl_atomic_load_relaxed(&call_cache[cache_idx[i]]); \
            if (entry && nargs == jl_svec_len(entry->sig->parameters) && \
                (entry->isleafsig ? sig_match_fast(FT, args, jl_svec_data(entry->sig->parameters), nargs) : \
                                    sig_match_any(FT, args, jl_svec_data(entry->sig->parameters), nargs)) && \
                world >= jl_atomic_load_relaxed(&entry->min_world) && world <= jl_atomic_load_relaxed(&entry->max_world)) { \
                goto have_entry; \
            } \
        } while (0);
    LOOP_BODY(0);
    LOOP_BODY(1);
    LOOP_BODY(2);
    LOOP_BODY(3);
#undef LOOP_BODY
    i = 4;
    if (i == 4) {
        // if no method was found in the associative cache, check the full cache
        JL_TIMING(METHOD_LOOKUP_FAST, METHOD_LOOKUP_FAST);
        jl_methcache_t *mc = jl_method_table->cache;
        jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
        entry = NULL;
        int cache_entry_count = jl_atomic_load_relaxed(&((jl_datatype_t*)FT)->name->cache_entry_count);
        if (leafcache != (jl_genericmemory_t*)jl_an_empty_memory_any && (cache_entry_count == 0 || cache_entry_count >= 8)) {
            // hashing args is expensive, but so do that only if looking at mc->cache is probably even more expensive
            tt = lookup_arg_type_tuple(F, args, nargs);
            if (tt != NULL)
                entry = lookup_leafcache(leafcache, (jl_value_t*)tt, world);
            dp_path = DP_LEAF;
        }
        if (entry == NULL) {
            dp_path = DP_ASSOC;
            jl_typemap_t *cache = jl_atomic_load_relaxed(&mc->cache); // XXX: gc root required?
            entry = jl_typemap_assoc_exact(cache, F, args, nargs, jl_cachearg_offset(), world);
            if (entry == NULL) {
                last_alloc = jl_options.malloc_log ? jl_gc_diff_total_bytes() : 0;
                if (tt == NULL) {
                    tt = arg_type_tuple(F, args, nargs);
                    // arg_type_tuple can allocate and hit a safepoint; a GC there may
                    // grow/replace mc->leafcache, freeing the memory `leafcache` still
                    // points at. Reload before dereferencing it in lookup_leafcache.
                    leafcache = jl_atomic_load_relaxed(&mc->leafcache);
                    entry = lookup_leafcache(leafcache, (jl_value_t*)tt, world);
                }
            }
        }
        if (entry != NULL && entry->isleafsig && entry->simplesig == (void*)jl_nothing && entry->guardsigs == jl_emptysvec) {
            // put the entry into the cache if it's valid for a leafsig lookup,
            // using pick_which to slightly randomize where it ends up
            // (intentionally not atomically synchronized, since we're just using it for randomness)
            // TODO: use the thread's `cong` instead as a source of randomness
            int which = jl_atomic_load_relaxed(&pick_which[cache_idx[0]]) + 1;
            jl_atomic_store_relaxed(&pick_which[cache_idx[0]], which);
            jl_atomic_store_release(&call_cache[cache_idx[which & 3]], entry);
        }
        if (entry) {
            // mfunc was found in slow path, so log --trace-dispatch
            jl_method_instance_t *mfunc = entry->func.linfo;
            record_dispatch_statement_on_first_dispatch(mfunc);
        }
        if (entry != NULL && call_cache_any_ok(entry)) {
            // one entry for every type at its `Any` positions: the cache entry itself, shared, so its
            // max_world stays open and invalidation lowers it (a polymorphic site does not thrash)
            int which = jl_atomic_load_relaxed(&pick_which[cache_idx[0]]) + 1;
            jl_atomic_store_relaxed(&pick_which[cache_idx[0]], which);
            jl_atomic_store_release(&call_cache[cache_idx[which & 3]], entry);
        }
    }

    jl_method_instance_t *mfunc;
    if (entry) {
have_entry:
        mfunc = entry->func.linfo;
    }
    else {
        assert(tt);
        // cache miss case
        dp_path = DP_FULL;
        jl_methcache_t *mc = jl_method_table->cache;
        if (jl_member_hook != NULL)
            member_hook_at_dispatch(tt, world);
        mfunc = jl_mt_assoc_by_type(jl_method_table, mc, tt, world);
        if (jl_options.malloc_log)
            jl_gc_sync_total_bytes(last_alloc); // discard allocation count from compilation
        if (mfunc == NULL) {
#ifdef JL_TRACE
            if (error_en)
                show_call(F, args, nargs);
#endif
            jl_method_error(F, args, nargs, world);
            // unreachable
        }
        // mfunc was found in slow path, so log --trace-dispatch
        record_dispatch_statement_on_first_dispatch(mfunc);
    }

#ifdef JL_TRACE
    if (traceen)
        jl_printf(JL_STDOUT, " at %s:%d\n", jl_symbol_name(mfunc->def.method->file), mfunc->def.method->line);
#endif
    if (__unlikely(dp_on != 0) && dp_enabled())
        dp_count(mfunc, dp_path);
    return mfunc;
}

// A dynamic call (compiler §2.4): the method lookup, with the per-call-site cache keyed by the
// caller's return address, then the code of the policy `reach` asks for. One copy, so that the
// lookup is inlined into it as into vanilla's jl_apply_generic; the entries below jump to it.
static NOINLINE jl_value_t *apply_generic_reach(jl_value_t *F, jl_value_t **args, uint32_t nargs, void *ip, int reach)
{
    size_t world = jl_current_task->world_age;
    if (__unlikely(dp_on > 0))
        dp_ip = ip;
    jl_method_instance_t *mfunc = jl_lookup_generic_(F, args, nargs,
                                                     jl_int32hash_fast(ip),
                                                     world);
    JL_GC_PROMISE_ROOTED(mfunc);
    return _jl_invoke(F, args, nargs, mfunc, world, reach);
}

// the runtime's own entry: the setting's default reaches
JL_DLLEXPORT jl_value_t *jl_apply_generic(jl_value_t *F, jl_value_t **args, uint32_t nargs)
{
    return apply_generic_reach(F, args, nargs, (void*)jl_return_address(), JL_REACH_RUNTIME);
}
// the entries of compiled code of each policy
JL_DLLEXPORT jl_value_t *jl_apply_generic_plain(jl_value_t *F, jl_value_t **args, uint32_t nargs)
{
    return apply_generic_reach(F, args, nargs, (void*)jl_return_address(), JL_REACH_PLAIN);
}
JL_DLLEXPORT jl_value_t *jl_apply_generic_facts(jl_value_t *F, jl_value_t **args, uint32_t nargs)
{
    return apply_generic_reach(F, args, nargs, (void*)jl_return_address(), JL_REACH_FACTS);
}
// a call made for code of a policy (a builtin that calls a function, the interpreter)
JL_DLLEXPORT jl_value_t *jl_apply_generic_reach(jl_value_t *F, jl_value_t **args, uint32_t nargs, int reach)
{
    return apply_generic_reach(F, args, nargs, (void*)jl_return_address(), reach);
}

static jl_method_match_t *_gf_invoke_lookup(jl_value_t *types JL_PROPAGATES_ROOT, jl_methtable_t *mt, size_t world, int cache_result, size_t *min_valid, size_t *max_valid)
{
    jl_value_t *unw = jl_unwrap_unionall((jl_value_t*)types);
    if (!jl_is_tuple_type(unw))
        return NULL;
    if (jl_tparam0(unw) == jl_bottom_type)
        return NULL;
    jl_methcache_t *mc = ((jl_methtable_t*)mt)->cache;
    jl_value_t *matches = ml_matches((jl_methtable_t*)mt, mc, (jl_tupletype_t*)types, 1, 0, 0, world, cache_result, min_valid, max_valid, NULL);
    if (matches == jl_nothing || jl_array_nrows(matches) != 1)
        return NULL;
    jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(matches, 0);
    return matc;
}

JL_DLLEXPORT jl_value_t *jl_gf_invoke_lookup(jl_value_t *types, jl_value_t *mt, size_t world)
{
    // Deprecated: Use jl_gf_invoke_lookup_worlds for future development
    size_t min_valid = 0;
    size_t max_valid = ~(size_t)0;
    if (mt == jl_nothing)
        mt = (jl_value_t*)jl_method_table;
    jl_method_match_t *matc = _gf_invoke_lookup(types, (jl_methtable_t*)mt, world, 1, &min_valid, &max_valid);
    if (matc == NULL)
        return jl_nothing;
    return (jl_value_t*)matc->method;
}


JL_DLLEXPORT jl_value_t *jl_gf_invoke_lookup_worlds(jl_value_t *types, jl_value_t *mt, size_t world, size_t *min_world, size_t *max_world)
{
    if (mt == jl_nothing)
        mt = (jl_value_t*)jl_method_table;
    jl_method_match_t *matc = _gf_invoke_lookup(types, (jl_methtable_t*)mt, world, 1, min_world, max_world);
    if (matc == NULL)
        return jl_nothing;
    return (jl_value_t*)matc;
}

// invoke()
// this does method dispatch with a set of types to match other than the
// types of the actual arguments. this means it sometimes does NOT call the
// most specific method for the argument types, so we need different logic.
// first we use the given types to look up a definition, then we perform
// caching and specialization within just that definition.
// every definition has its own private method table for this purpose.
//
// NOTE: assumes argument type is a subtype of the lookup type.
jl_value_t *jl_gf_invoke(jl_value_t *types0, jl_value_t *gf, jl_value_t **args, size_t nargs)
{
    return jl_gf_invoke_reach(types0, gf, args, nargs, JL_REACH_RUNTIME);
}

jl_value_t *jl_gf_invoke_reach(jl_value_t *types0, jl_value_t *gf, jl_value_t **args, size_t nargs, int reach)
{
    size_t world = jl_current_task->world_age;
    jl_value_t *types = NULL;
    JL_GC_PUSH1(&types);
    types = jl_argtype_with_function((jl_value_t*)gf, types0);
    jl_method_t *method = (jl_method_t*)jl_gf_invoke_lookup(types, jl_nothing, world);
    JL_GC_PROMISE_ROOTED(method);

    if ((jl_value_t*)method == jl_nothing) {
        jl_method_error_bare(gf, types0, world);
        // unreachable
    }

    // now we have found the matching definition.
    // next look for or create a specialization of this definition.
    JL_GC_POP();
    return jl_gf_invoke_by_method_reach(method, gf, args, nargs, reach);
}

jl_value_t *jl_gf_invoke_by_method(jl_method_t *method, jl_value_t *gf, jl_value_t **args, size_t nargs)
{
    return jl_gf_invoke_by_method_reach(method, gf, args, nargs, JL_REACH_RUNTIME);
}

jl_value_t *jl_gf_invoke_by_method_reach(jl_method_t *method, jl_value_t *gf, jl_value_t **args, size_t nargs, int reach)
{
    jl_method_instance_t *mfunc = NULL;
    jl_typemap_entry_t *tm = NULL;
    jl_typemap_t *invokes = jl_atomic_load_relaxed(&method->invokes);
    if (invokes != jl_nothing)
        tm = jl_typemap_assoc_exact(invokes, gf, args, nargs, 1, 1);
    if (tm) {
        mfunc = tm->func.linfo;
    }
    else {
        int64_t last_alloc = jl_options.malloc_log ? jl_gc_diff_total_bytes() : 0;
        jl_svec_t *tpenv = jl_emptysvec;
        jl_tupletype_t *tt = NULL;
        JL_GC_PUSH2(&tpenv, &tt);
        JL_LOCK(&method->writelock);
        invokes = jl_atomic_load_relaxed(&method->invokes);
        tm = jl_typemap_assoc_exact(invokes, gf, args, nargs, 1, 1);
        if (tm) {
            mfunc = tm->func.linfo;
        }
        else {
            tt = arg_type_tuple(gf, args, nargs);
            if (jl_is_unionall(method->sig)) {
                int sub = jl_subtype_matching((jl_value_t*)tt, (jl_value_t*)method->sig, &tpenv);
                assert(sub); (void)sub;
            }
            mfunc = cache_method(NULL, NULL, &method->invokes, (jl_value_t*)method, tt, method, 1, 1, ~(size_t)0, tpenv, /*tt_known_absent*/0);
        }
        JL_UNLOCK(&method->writelock);
        JL_GC_POP();
        if (jl_options.malloc_log)
            jl_gc_sync_total_bytes(last_alloc); // discard allocation count from compilation
    }
    JL_GC_PROMISE_ROOTED(mfunc);
    uint8_t force_trace_dispatch = jl_atomic_load_relaxed(&jl_force_trace_dispatch_enabled);
    if (force_trace_dispatch || jl_options.trace_dispatch != NULL) {
        uint8_t miflags = jl_atomic_load_relaxed(&mfunc->flags);
        uint8_t was_dispatched = miflags & JL_MI_FLAGS_MASK_DISPATCHED;
        if (!was_dispatched) {
            miflags |= JL_MI_FLAGS_MASK_DISPATCHED;
            jl_atomic_store_relaxed(&mfunc->flags, miflags);
            record_dispatch_statement(mfunc);
        }
    }
    size_t world = jl_current_task->world_age;
    return _jl_invoke(gf, args, nargs - 1, mfunc, world, reach);
}

jl_sym_t *jl_gf_supertype_name(jl_sym_t *name)
{
    size_t l = strlen(jl_symbol_name(name));
    char *prefixed;
    prefixed = (char*)malloc_s(l+2);
    prefixed[0] = '#';
    strcpy(&prefixed[1], jl_symbol_name(name));
    jl_sym_t *tname = jl_symbol(prefixed);
    free(prefixed);
    return tname;
}

// Return value is rooted globally
jl_value_t *jl_new_generic_function_with_supertype(jl_sym_t *name, jl_module_t *module, jl_datatype_t *st, size_t new_world)
{
    // type name is function name prefixed with #
    jl_sym_t *tname = jl_gf_supertype_name(name);
    jl_datatype_t *ftype = (jl_datatype_t*)jl_new_datatype(
            tname, module, st, jl_emptysvec, jl_emptysvec, jl_emptysvec, jl_emptysvec,
            0, 0, 0);
    assert(jl_is_datatype(ftype));
    JL_GC_PUSH1(&ftype);
    ftype->name->singletonname = name;
    jl_gc_wb(ftype->name, name);
    jl_declare_constant_val3(NULL, module, tname, (jl_value_t*)ftype, PARTITION_KIND_CONST, new_world);
    jl_value_t *f = jl_new_struct(ftype);
    ftype->instance = f;
    jl_gc_wb(ftype, f);
    JL_GC_POP();
    return (jl_value_t*)f;
}

jl_value_t *jl_new_generic_function(jl_sym_t *name, jl_module_t *module, size_t new_world)
{
    return jl_new_generic_function_with_supertype(name, module, jl_function_type, new_world);
}

struct ml_matches_env {
    // inputs:
    struct typemap_intersection_env match;
    int intersections;
    size_t world;
    int lim;
    int include_ambiguous;
    // results:
    jl_value_t *t; // array of method matches
    jl_method_match_t *matc; // current working method match
};

enum SIGNATURE_FULLY_COVERS {
    NOT_FULLY_COVERS = 0,
    FULLY_COVERS = 1,
    SENTINEL    = 2,
};

static jl_method_match_t *make_method_match(jl_tupletype_t *spec_types, jl_svec_t *sparams, jl_method_t *method, enum SIGNATURE_FULLY_COVERS fully_covers)
{
    jl_task_t *ct = jl_current_task;
    jl_method_match_t *match = (jl_method_match_t*)jl_gc_alloc(ct->ptls, sizeof(jl_method_match_t), jl_method_match_type);
    match->spec_types = spec_types;
    match->sparams = sparams;
    match->method = method;
    match->fully_covers = fully_covers;
    return match;
}

// callback for typemap_visitor
//
// This will exit the search early (by returning 0 / false) if the match limit is proven to be
// exceeded early. This is only best-effort, since specificity means that many matched methods
// may be sorted and removed in the output processing for ml_matches and therefore we can only
// conservatively under-approximate the matches during the search.
static int ml_matches_visitor(jl_typemap_entry_t *ml, struct typemap_intersection_env *closure0)
{
    struct ml_matches_env *closure = container_of(closure0, struct ml_matches_env, match);
    if (closure->intersections == 0 && !closure0->issubty)
        return 1;

    // First, check the world range of the typemap entry to ensure that it intersects
    // the query world. If it does not, narrow the result world range to guarantee
    // excluding it from the results is valid for the full span.
    size_t min_world = jl_atomic_load_relaxed(&ml->min_world);
    size_t max_world = jl_atomic_load_relaxed(&ml->max_world);
    if (closure->world < min_world) {
        // exclude method table entries that are part of a later world
        if (closure->match.max_valid >= min_world)
            closure->match.max_valid = min_world - 1;
        return 1;
    }
    else if (closure->world > max_world) {
        // exclude method table entries that have been replaced in the current world
        if (closure->match.min_valid <= max_world)
            closure->match.min_valid = max_world + 1;
        return 1;
    }
    if (closure->match.max_valid > max_world)
        closure->match.max_valid = max_world;
    jl_method_t *meth = ml->func.method;
    int only = jl_atomic_load_relaxed(&meth->dispatch_status) & METHOD_SIG_LATEST_ONLY;
    if (closure->lim >= 0 && only) {
        if (closure->lim == 0) {
            closure->t = jl_an_empty_vec_any;
            return 0;
        }
        closure->lim--;
    }
    // a match that holds only through membership entries is valid only while
    // those entries live (I2: this is the dispatch-side world bound — inference
    // intersects it into sv.world.valid_worlds the way it does method worlds)
    if (sig_has_open_set((jl_value_t*)ml->sig)) {
        size_t lo = 1, hi = ~(size_t)0;
        if (jl_membership_world_range(closure->match.ti, closure->world, &lo, &hi)) {
            if (closure->match.min_valid < lo)
                closure->match.min_valid = lo;
            if (closure->match.max_valid > hi)
                closure->match.max_valid = hi;
        }
    }
    closure->matc = make_method_match((jl_tupletype_t*)closure->match.ti,
        closure->match.env, meth,
        closure->match.issubty ? FULLY_COVERS : NOT_FULLY_COVERS);
    size_t len = jl_array_nrows(closure->t);
    if (closure->match.issubty && only) {
        if (len == 0)
            closure->t = (jl_value_t*)jl_alloc_vec_any(1);
        else if (len > 1)
            jl_array_del_end((jl_array_t*)closure->t, len - 1);
        jl_array_ptr_set(closure->t, 0, (jl_value_t*)closure->matc);
        return 0;
    }
    if (len == 0) {
        closure->t = (jl_value_t*)jl_alloc_vec_any(1);
        jl_array_ptr_set(closure->t, 0, (jl_value_t*)closure->matc);
    }
    else {
        jl_array_ptr_1d_push((jl_array_t*)closure->t, (jl_value_t*)closure->matc);
    }
    // don't need to consider other similar methods if this ml will always fully intersect with them and dominates all of them
    if (!closure->include_ambiguous || closure->lim != -1)
        typemap_slurp_search(ml, &closure->match);
    return 1;
}

// Visit the candidate methods, starting from t[idx], to determine a possible valid sort ordering,
// where every morespecific method appears before any method which it has a common
// intersection with but is not partly ambiguous with (ambiguity is not transitive, since morespecific is not transitive).
// Implements Tarjan's SCC (strongly connected components) algorithm, simplified to remove the count variable
// Inputs:
//  * `t`: the array of vertexes (method matches)
//  * `idx`: the next vertex to add to the output
//  * `visited`: the state of the algorithm for each vertex in `t`: either 1 if we visited it already or 1+depth if we are visiting it now
//  * `stack`: the state of the algorithm for the current vertex (up to length equal to `t`): the list of all vertexes currently in the depth-first path or in the current SCC
//  * `result`: the output of the algorithm, a sorted list of vertexes (up to length `lim`)
//  * `recursion_stack`: an array for temporary use
//  * `lim`: either -1 for unlimited matches, or the maximum length for `result` before returning failure (return -1).
//  * `include_ambiguous`: whether to filter out fully ambiguous matches from `result`
//  * `*has_ambiguity`: whether the algorithm does not need to compute if there is an unresolved ambiguity
//  * `*found_minmax`: whether there is a minmax method already found, so future fully_covers matches should be ignored
// Outputs:
//  * `*has_ambiguity`: whether there are any ambiguities that mean the sort order is not exact
// Stack frame for iterative sort_mlmatches implementation
enum sort_state {
    STATE_VISITING,            // Initial visit and setup
    STATE_PROCESSING_INTERFERENCES, // Processing interference loop
    STATE_CHECK_COVERS,        // Check coverage conditions
    STATE_FINALIZE_SCC         // SCC processing and cleanup
};

typedef struct {
    size_t idx;                    // Current method match index
    size_t interference_index;     // Current position in interferences loop
    size_t interference_count;     // Total interferences count
    size_t depth;                  // Stack depth when frame created
    size_t cycle;                  // Cycle depth tracking
    jl_method_match_t *matc;       // Current method match
    jl_method_t *m;                // Current method
    jl_value_t *ti;                // Type intersection
    int subt;                      // Subtype flag
    jl_genericmemory_t *interferences; // Method interferences
    int child_result;              // Result from child recursive call
    enum sort_state state;
} sort_stack_frame_t;

// Returns:
//  * -1: too many matches for lim, other outputs are undefined
//  *  0: the child(ren) have been added to the output
//  * 1+: the children are part of this SCC (up to this depth)
static int sort_mlmatches(jl_array_t *t, size_t idx, arraylist_t *visited, arraylist_t *stack, arraylist_t *result, arraylist_t *recursion_stack, int lim, int include_ambiguous, int *has_ambiguity, int *found_minmax)
{
    // Use arraylist_t for explicit stack of processing frames
    arraylist_t frame_stack;
    arraylist_new(&frame_stack, 0);

    // Push initial frame
    sort_stack_frame_t initial_frame = {
        .idx = idx,
        .interference_index = 0,
        .interference_count = 0,
        .depth = 0,
        .cycle = 0,
        .matc = NULL,
        .m = NULL,
        .ti = NULL,
        .subt = 0,
        .interferences = NULL,
        .child_result = 0,
        .state = STATE_VISITING
    };
    arraylist_push(&frame_stack, memcpy(malloc(sizeof(sort_stack_frame_t)), &initial_frame, sizeof(sort_stack_frame_t)));

    int final_result = 0;

    while (1) {
        sort_stack_frame_t *current = (sort_stack_frame_t*)frame_stack.items[frame_stack.len - 1];
        JL_GC_PROMISE_ROOTED(current->m);
        JL_GC_PROMISE_ROOTED(current->interferences);
        JL_GC_PROMISE_ROOTED(current->ti);

        switch (current->state) {
            case STATE_VISITING: {
                size_t cycle = (size_t)visited->items[current->idx];
                if (cycle != 0) {
                    final_result = cycle - 1;
                    goto propagate_to_parent;
                }

                arraylist_push(stack, (void*)current->idx);
                current->depth = stack->len;
                visited->items[current->idx] = (void*)(1 + current->depth);
                current->matc = (jl_method_match_t*)jl_array_ptr_ref(t, current->idx);
                current->m = current->matc->method;
                current->ti = (jl_value_t*)current->matc->spec_types;
                current->subt = current->matc->fully_covers != NOT_FULLY_COVERS;
                current->interferences = jl_atomic_load_relaxed(&current->m->interferences);
                current->cycle = current->depth;
                current->interference_count = current->interferences->length;
                current->interference_index = 0;
                current->state = STATE_PROCESSING_INTERFERENCES;
                break;
            }

            case STATE_PROCESSING_INTERFERENCES: {
                // If we have a child result to process, handle it first
                if (current->child_result != 0) {
                    if (current->child_result == -1) {
                        final_result = -1;
                        goto propagate_to_parent;
                    }
                    // record the cycle will resolve at depth "cycle"
                    if (current->child_result && current->child_result < current->cycle)
                        current->cycle = current->child_result;
                    current->child_result = 0; // Clear after processing
                }

                // Process interferences iteratively
                while (current->interference_index < current->interference_count) {
                    jl_method_t *m2 = (jl_method_t*)jl_genericmemory_ptr_ref(current->interferences, current->interference_index);
                    current->interference_index++;

                    if (m2 == NULL)
                        continue;

                    int childidx = find_method_in_matches(t, m2);
                    if (childidx < 0 || (size_t)childidx == current->idx)
                        continue;

                    int child_cycle = (size_t)visited->items[childidx];
                    if (child_cycle == 1)
                        continue; // already handled
                    if (child_cycle != 0 && child_cycle - 1 >= current->cycle)
                        continue; // already part of this cycle
                    if (method_in_interferences(current->m, m2))
                        continue;

                    // m2 is morespecific, so attempt to visit it first
                    if (child_cycle != 0) {
                        // Child already being processed, use cached result
                        int child_result = child_cycle - 1;
                        if (child_result == -1) {
                            final_result = -1;
                            goto propagate_to_parent;
                        }
                        if (child_result && child_result < current->cycle)
                            current->cycle = child_result;
                    }
                    else {
                        // Need to process child - push new frame and pause current processing
                        sort_stack_frame_t child_frame = {
                            .idx = childidx,
                            .interference_index = 0,
                            .interference_count = 0,
                            .depth = 0,
                            .cycle = 0,
                            .matc = NULL,
                            .m = NULL,
                            .ti = NULL,
                            .subt = 0,
                            .interferences = NULL,
                            .child_result = 0,
                            .state = STATE_VISITING
                        };
                        arraylist_push(&frame_stack, memcpy(malloc(sizeof(sort_stack_frame_t)), &child_frame, sizeof(sort_stack_frame_t)));
                        goto continue_main_loop; // Resume processing after child completes
                    }
                }

                current->state = STATE_CHECK_COVERS;
                break;
            }

            case STATE_CHECK_COVERS: {
                // There is some probability that this method is already fully covered
                // now, and we can delete this vertex now without anyone noticing.
                if (current->subt && *found_minmax) {
                    if (*found_minmax == 2)
                        visited->items[current->idx] = (void*)1;
                }
                else if (check_interferences_covers(current->m, current->ti, t, visited, recursion_stack)) {
                    visited->items[current->idx] = (void*)1;
                }
                else if (check_fully_ambiguous(current->m, current->ti, t, include_ambiguous, has_ambiguity)) {
                    visited->items[current->idx] = (void*)1;
                }

                // If there were no cycles hit either, then we can potentially delete all of its edges too.
                if ((size_t)visited->items[current->idx] == 1 && stack->len == current->depth) {
                    // n.b. cycle might be < depth, if we had a cycle with a child
                    // idx, but since we are on the top of the stack, nobody
                    // observed that and so we are content to ignore this
                    size_t childidx = (size_t)arraylist_pop(stack);
                    assert(childidx == current->idx); (void)childidx;
                    final_result = 0;
                    goto propagate_to_parent;
                }

                if (current->cycle != current->depth) {
                    final_result = current->cycle;
                    goto propagate_to_parent;
                }

                current->state = STATE_FINALIZE_SCC;
                break;
            }

            case STATE_FINALIZE_SCC: {
                // If this is in an SCC group, do some additional checks before returning or setting has_ambiguity
                if (current->depth != stack->len) {
                    int scc_count = 0;
                    for (size_t i = current->depth - 1; i < stack->len; i++) {
                        size_t childidx = (size_t)stack->items[i];
                        if (visited->items[childidx] == (void*)1)
                            continue;
                        scc_count++;
                    }
                    if (scc_count > 1)
                        *has_ambiguity = 1;
                }

                // copy this cycle into the results
                for (size_t i = current->depth - 1; i < stack->len; i++) {
                    size_t childidx = (size_t)stack->items[i];
                    jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(t, childidx);
                    int subt = matc->fully_covers != NOT_FULLY_COVERS;
                    if (subt && *found_minmax)
                        visited->items[childidx] = (void*)1;
                    if ((size_t)visited->items[childidx] == 1)
                        continue;
                    assert(visited->items[childidx] == (void*)(2 + i));
                    visited->items[childidx] = (void*)1;
                    if (lim == -1 || result->len < lim)
                        arraylist_push(result, (void*)childidx);
                    else {
                        final_result = -1;
                        goto propagate_to_parent;
                    }
                }

                // now finally cleanup the stack
                while (stack->len >= current->depth) {
                    size_t childidx = (size_t)arraylist_pop(stack);
                    // always remove fully_covers matches after the first minmax ambiguity group is handled
                    jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(t, childidx);
                    int subt = matc->fully_covers == FULLY_COVERS;
                    if (subt && *found_minmax == 1)
                        *found_minmax = 2;
                    assert(visited->items[childidx] == (void*)1);
                }

                final_result = 0;
                goto propagate_to_parent;
            }
        }

        continue_main_loop:
            continue;

        propagate_to_parent:
            // Propagate result to parent if exists
            free(arraylist_pop(&frame_stack));
            if (frame_stack.len == 0)
                break;
            sort_stack_frame_t *parent = (sort_stack_frame_t*)frame_stack.items[frame_stack.len - 1];
            parent->child_result = final_result;
    }
    assert(frame_stack.len == 0);
    arraylist_free(&frame_stack);
    return final_result;
}


// This is the collect form of calling jl_typemap_intersection_visitor
// with optimizations to skip fully shadowed methods.
//
// Returns a match as an array of svec(argtypes, static_params, Method, fully-covers).
//
// See below for the meaning of lim.
//
// fully-covers is a Bool indicating subtyping, though temporarily it may be
// tri-values, with `nothing` indicating a match that is not a subtype, but
// which is dominated by one which is (and thus should be excluded unless ambiguous)
static jl_value_t *ml_matches_i(jl_methtable_t *mt, jl_methcache_t *mc,
                              jl_tupletype_t *type, int lim, int include_ambiguous,
                              int intersections, size_t world, int cache_result,
                              size_t *min_valid, size_t *max_valid, int *ambig)
{
    if (world > jl_atomic_load_acquire(&jl_world_counter))
        return jl_nothing; // the future is not enumerable
    JL_TIMING(METHOD_MATCH, METHOD_MATCH);
    int has_ambiguity = 0;
    jl_value_t *unw = jl_unwrap_unionall((jl_value_t*)type);
    assert(jl_is_datatype(unw));
    size_t l = jl_svec_len(((jl_datatype_t*)unw)->parameters);
    jl_value_t *va = NULL;
    if (l > 0) {
        va = jl_tparam(unw, l - 1);
        if (jl_is_vararg(va))
            va = jl_unwrap_vararg(va);
        else
            va = NULL;
    }
    struct ml_matches_env env = {
        /* match */ {
            /* inputs */
            /* fptr / callback */ ml_matches_visitor,
            /* sig */ (jl_value_t*)type,
            /* vararg type / tparam0 */ va,

            /* temporaries */
            /* .search_slurp = */ 0,
            /* .world = */ world,

            /* outputs */
            /* .min_valid = */ *min_valid,
            /* .max_valid = */ *max_valid,
            /* .ti = */ NULL,
            /* .env = */ jl_emptysvec,
            /* .issubty = */ 0
        },
        /* inputs */
        intersections,
        world,
        lim,
        include_ambiguous,

        /* outputs */
        /* .t = */ jl_an_empty_vec_any,

        /* temporaries */
        /* .matc = */ NULL
    };
    struct jl_typemap_assoc search = {(jl_value_t*)type, world, jl_emptysvec};
    jl_value_t *isect2 = NULL;
    JL_GC_PUSH6(&env.t, &env.matc, &env.match.env, &search.env, &env.match.ti, &isect2);

    if (mc) {
        // first check the leaf cache if the type might have been put in there
        if (((jl_datatype_t*)unw)->isdispatchtuple) {
            jl_genericmemory_t *leafcache = jl_atomic_load_relaxed(&mc->leafcache);
            jl_typemap_entry_t *entry = lookup_leafcache(leafcache, (jl_value_t*)type, world);
            if (entry) {
                // leafcache found a match, construct the MethodMatch by computing the effective
                // types + sparams and the world bounds
                jl_method_instance_t *mi = entry->func.linfo;
                jl_method_t *meth = mi->def.method;
                if (!jl_is_unionall(meth->sig)) {
                    env.match.env = jl_emptysvec;
                    env.match.ti = unw;
                }
                else if (jl_egal((jl_value_t*)type, mi->specTypes)) {
                    env.match.env = mi->sparam_vals;
                    env.match.ti = mi->specTypes;
                }
                else {
                    // this just calls jl_subtype_env (since we know that `type <: meth->sig` by transitivity)
                    env.match.ti = jl_type_intersection_env((jl_value_t*)type, (jl_value_t*)meth->sig, &env.match.env);
                }
                env.matc = make_method_match((jl_tupletype_t*)env.match.ti,
                    env.match.env, meth, FULLY_COVERS);
                env.t = (jl_value_t*)jl_alloc_vec_any(1);
                jl_array_ptr_set(env.t, 0, env.matc);
                size_t min_world = jl_atomic_load_relaxed(&entry->min_world);
                size_t max_world = jl_atomic_load_relaxed(&entry->max_world);
                if (*min_valid < min_world)
                    *min_valid = min_world;
                if (*max_valid > max_world)
                    *max_valid = max_world;
                JL_GC_POP();
                return env.t;
            }
        }

        // then check the full cache if it seems profitable
        if (((jl_datatype_t*)unw)->isdispatchtuple) {
            jl_typemap_entry_t *entry = jl_typemap_assoc_by_type(jl_atomic_load_relaxed(&mc->cache), &search, jl_cachearg_offset(), /*subtype*/1);
            if (entry && (((jl_datatype_t*)unw)->isdispatchtuple || entry->guardsigs == jl_emptysvec)) {
                // full cache found a match, construct the MethodMatch by computing the effective
                // types + sparams and the world bounds
                jl_method_instance_t *mi = entry->func.linfo;
                jl_method_t *meth = mi->def.method;
                size_t min_world = jl_atomic_load_relaxed(&entry->min_world);
                // only return this if it appears min_would is fully computed, otherwise do the full lookup to compute min_world exactly
                if (min_world == jl_atomic_load_relaxed(&meth->primary_world)) {
                    size_t max_world = jl_atomic_load_relaxed(&entry->max_world);
                    if (!jl_is_unionall(meth->sig) && ((jl_datatype_t*)unw)->isdispatchtuple) {
                        env.match.env = jl_emptysvec;
                        env.match.ti = unw;
                    }
                    else {
                        // this just calls jl_subtype_env (since we know that `type <: meth->sig` by transitivity)
                        env.match.ti = jl_type_intersection_env((jl_value_t*)type, (jl_value_t*)meth->sig, &env.match.env);
                    }
                    env.matc = make_method_match((jl_tupletype_t*)env.match.ti,
                        env.match.env, meth, FULLY_COVERS);
                    env.t = (jl_value_t*)jl_alloc_vec_any(1);
                    jl_array_ptr_set(env.t, 0, env.matc);
                    if (*min_valid < min_world)
                        *min_valid = min_world;
                    if (*max_valid > max_world)
                        *max_valid = max_world;
                    JL_GC_POP();
                    return env.t;
                }
            }
        }
    }
    // then scan everything
    if (!jl_typemap_intersection_visitor(jl_atomic_load_relaxed(&mt->defs), 0, &env.match) && env.t == jl_an_empty_vec_any) {
        JL_GC_POP();
        // if we return early without returning methods, lim was proven to be exceeded
        // during the search set only the min/max valid collected from matching
        *min_valid = env.match.min_valid;
        *max_valid = env.match.max_valid;
        return jl_nothing;
    }
    // if we return early, set only the min/max valid collected from matching
    *min_valid = env.match.min_valid;
    *max_valid = env.match.max_valid;
    // done with many of these values now
    env.match.ti = NULL; env.matc = NULL; env.match.env = NULL; search.env = NULL;

    // all intersecting methods have been collected now. the remaining work is to sort
    // these and apply specificity to determine a list of dispatch-possible call targets
    size_t i, j, len = jl_array_nrows(env.t);

    // the 'minmax' method is a method that (1) fully-covers the queried type, and (2) is
    // more-specific than any other fully-covering method (but if !all_subtypes, there are
    // non-fully-covering methods to which it is _likely_ not more specific)
    jl_method_match_t *minmax = NULL;
    int any_subtypes = 0;
    if (len > 1) {
        // first try to pre-process the results to find the most specific option
        // among the fully-covering methods, since we can do this in O(n^2)
        // time, and the rest is O(n^3)
        //   - first find a candidate for the best of these method results
        for (i = 0; i < len; i++) {
            jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(env.t, i);
            if (matc->fully_covers == FULLY_COVERS) {
                any_subtypes = 1;
                jl_method_t *m = matc->method;
                for (j = 0; j < len; j++) {
                    if (i == j)
                        continue;
                    jl_method_match_t *matc2 = (jl_method_match_t*)jl_array_ptr_ref(env.t, j);
                    if (matc2->fully_covers == FULLY_COVERS) {
                        jl_method_t *m2 = matc2->method;
                        if (!method_morespecific_via_interferences(m, m2))
                            break;
                    }
                }
                if (j == len) {
                    // Found the minmax method
                    minmax = matc;
                    break;
                }
            }
        }
        //   - it may even dominate (be more specific than) some choices that are not fully-covering!
        //     move those into the subtype group, where we'll filter them out shortly after
        //     (potentially avoiding reporting these as an ambiguity, and
        //     potentially allowing us to hit the next fast path)
        //   - we could always check here if *any* FULLY_COVERS method is
        //     more-specific (instead of just considering minmax), but that may
        //     cost much extra and is less likely to help us hit a fast path
        //     (we will look for this later, when we compute ambig_groupid, for
        //     correctness)
        int all_subtypes = any_subtypes;
        if (any_subtypes) {
            jl_method_t *minmaxm = NULL;
            if (minmax != NULL)
                minmaxm = minmax->method;
            // scan through all the non-fully-matching methods and count them as "fully-covering" (ish)
            // (i.e. in the 'subtype' group) if `minmax` is more-specific
            for (i = 0; i < len; i++) {
                jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(env.t, i);
                if (matc->fully_covers != FULLY_COVERS) {
                    jl_method_t *m = matc->method;
                    if (minmaxm) {
                        if (method_morespecific_via_interferences(minmaxm, m)) {
                            matc->fully_covers = SENTINEL; // put a sentinel value here for sorting
                            continue;
                        }
                        if (method_in_interferences(minmaxm, m)) // !morespecific(m, minmaxm)
                            has_ambiguity = 1;
                    }
                    all_subtypes = 0;
                }
            }
        }
        //    - now we might have a fast-return here, if we see that
        //      we've already processed all of the possible outputs
        if (all_subtypes) {
            if (minmax == NULL) {
                // all intersecting methods are fully-covering, but there is no unique most-specific method
                if (!include_ambiguous) {
                    // there no unambiguous choice of method
                    len = 0;
                    env.t = jl_an_empty_vec_any;
                }
                else if (lim == 1) {
                    // we'd have to return >1 method due to the ambiguity, so bail early
                    JL_GC_POP();
                    return jl_nothing;
                }
            }
            else {
                // `minmax` is more-specific than all other matches and is fully-covering
                // we can return it as our only result
                jl_array_ptr_set(env.t, 0, minmax);
                jl_array_del_end((jl_array_t*)env.t, len - 1);
                len = 1;
            }
        }
        if (minmax && lim == 0) {
            // protect some later algorithms from underflow
            JL_GC_POP();
            return jl_nothing;
        }
    }
    if (len > 1) {
        arraylist_t stack, visited, result, recursion_stack;
        arraylist_new(&result, lim != -1 && lim < len ? lim : len);
        arraylist_new(&stack, 0);
        arraylist_new(&visited, len);
        arraylist_new(&recursion_stack, len);
        arraylist_grow(&visited, len);
        memset(visited.items, 0, len * sizeof(size_t));
        // if we had a minmax method (any subtypes), now may now be able to
        // quickly cleanup some of methods
        int found_minmax = 0;
        if (has_ambiguity)
            found_minmax = 1;
        else if (minmax != NULL)
            found_minmax = 2;
        else if (any_subtypes && !include_ambiguous)
            found_minmax = 1;
        has_ambiguity = 0;
        if (ambig == NULL) // if we don't care about the result, set it now so we won't bother attempting to compute it accurately later
            has_ambiguity = 1;
        for (i = 0; i < len; i++) {
            assert(visited.items[i] == (void*)0 || visited.items[i] == (void*)1);
            jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(env.t, i);
            if (matc->fully_covers != NOT_FULLY_COVERS && found_minmax) {
                // this was already handled above and below, so we won't learn anything new
                // by visiting it and it might be a bit costly
                continue;
            }
            int child_cycle = sort_mlmatches((jl_array_t*)env.t, i, &visited, &stack, &result, &recursion_stack, lim == -1 || minmax == NULL ? lim : lim - 1, include_ambiguous, &has_ambiguity, &found_minmax);
            if (child_cycle == -1) {
                arraylist_free(&recursion_stack);
                arraylist_free(&visited);
                arraylist_free(&stack);
                arraylist_free(&result);
                JL_GC_POP();
                return jl_nothing;
            }
            assert(child_cycle == 0); (void)child_cycle;
            assert(stack.len == 0);
            assert(visited.items[i] == (void*)1);
        }
        arraylist_free(&recursion_stack);
        arraylist_free(&visited);
        arraylist_free(&stack);
        for (j = 0; j < result.len; j++) {
            i = (size_t)result.items[j];
            jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(env.t, i);
            // remove our sentinel entry markers
            if (matc->fully_covers == SENTINEL)
                matc->fully_covers = NOT_FULLY_COVERS;
            result.items[j] = (void*)matc;
        }
        if (minmax) {
            arraylist_push(&result, minmax);
            j++;
        }
        memcpy(jl_array_data(env.t, jl_method_match_t*), result.items, j * sizeof(jl_method_match_t*));
        arraylist_free(&result);
        if (j != len)
            jl_array_del_end((jl_array_t*)env.t, len - j);
        len = j;
    }
    for (j = 0; j < len; j++) {
        jl_method_match_t *matc = (jl_method_match_t*)jl_array_ptr_ref(env.t, j);
        jl_method_t *m = matc->method;
        // method applicability is the same as typemapentry applicability
        size_t min_world = jl_atomic_load_relaxed(&m->primary_world);
        // intersect the env valid range with method lookup's inclusive valid range
        if (env.match.min_valid < min_world)
            env.match.min_valid = min_world;
    }
    if (mc && cache_result && ((jl_datatype_t*)unw)->isdispatchtuple) { // cache_result parameter keeps this from being recursive
        if (len == 1 && !has_ambiguity) {
            env.matc = (jl_method_match_t*)jl_array_ptr_ref(env.t, 0);
            jl_method_t *meth = env.matc->method;
            jl_svec_t *tpenv = env.matc->sparams;
            JL_LOCK(&mc->writelock);
            // a matching entry may already be present here (e.g. ml_matches fell
            // through the full-cache check above to recompute min_world exactly),
            // so cache_method must keep its own presence check to avoid a
            // duplicate insertion
            cache_method(mt, mc, &mc->cache, (jl_value_t*)mc, (jl_tupletype_t*)unw, meth, world, env.match.min_valid, env.match.max_valid, tpenv, /*tt_known_absent*/0);
        }
    }
    *min_valid = env.match.min_valid;
    *max_valid = env.match.max_valid;
    if (ambig != NULL)
        *ambig = has_ambiguity;
    JL_GC_POP();
    if (lim >= 0 && len > lim)
        return jl_nothing;
    return env.t;
}

// §8 (exploration §2.3, review M10): the order-independent ambiguity report at a
// membership declaration. A member of X and Y with a function holding a method
// at both and nothing more specific is a real ambiguity at that member: Julia's
// own dispatch raises it at the call; we warn now, at the declaration, where the
// whole picture exists. WARN-ONLY contract (W4-10): this runs under
// world_counter_lock — nothing here may throw; jl_printf/jl_static_show only.
static void membership_ambig_warn(jl_method_t *m, jl_value_t *X, jl_value_t *Y,
                                  jl_value_t *member, jl_value_t *joint)
{
    jl_printf(JL_STDERR, "WARNING: method %s", jl_symbol_name(m->name));
    jl_printf(JL_STDERR, " at ");
    jl_static_show(JL_STDERR, X);
    jl_printf(JL_STDERR, " and ");
    jl_static_show(JL_STDERR, Y);
    jl_printf(JL_STDERR, " is ambiguous for ");
    jl_static_show(JL_STDERR, member);
    jl_printf(JL_STDERR, ", a member of both sets");
    if (joint) {
        jl_printf(JL_STDERR, "; a method at ");
        jl_static_show(JL_STDERR, joint);
        jl_printf(JL_STDERR, " would resolve it");
    }
    jl_printf(JL_STDERR, "\n");
}

// `sig` with `member` in place of each argument typed by an instance of the set `tn`, also
// inside a Vararg; NULL where no argument is. The method's other arguments stay as they are,
// so the query has the method's own arity (compiler §5.4).
static jl_value_t *sig_at_member(jl_value_t *sig, jl_typename_t *tn, jl_value_t *member)
{
    jl_value_t *body = jl_unwrap_unionall(sig);
    if (!jl_is_tuple_type(body))
        return NULL;
    size_t n = jl_nparams((jl_datatype_t*)body);
    jl_svec_t *ps = NULL;
    jl_value_t *res = NULL;
    JL_GC_PUSH2(&ps, &res);
    ps = jl_svec_copy(((jl_datatype_t*)body)->parameters);
    int any = 0;
    for (size_t i = 1; i < n; i++) {
        jl_value_t *p = jl_svecref(ps, i);
        if (jl_is_vararg(p)) {
            jl_vararg_t *va = (jl_vararg_t*)p;
            jl_value_t *t = va->T ? jl_unwrap_unionall(va->T) : NULL;
            if (t && jl_is_datatype(t) && ((jl_datatype_t*)t)->name == tn) {
                res = (jl_value_t*)jl_wrap_vararg(member, va->N, 1, 0);
                jl_svecset(ps, i, res);
                any = 1;
            }
            continue;
        }
        jl_value_t *pu = jl_unwrap_unionall(p);
        if (jl_is_datatype(pu) && ((jl_datatype_t*)pu)->name == tn) {
            jl_svecset(ps, i, member);
            any = 1;
        }
    }
    res = NULL;
    if (any) {
        res = (jl_value_t*)jl_apply_tuple_type(ps, 1);
        res = jl_rewrap_unionall(res, sig);
    }
    JL_GC_POP();
    return res;
}

// whether mX and mY are ambiguous at the call types `q` (the member in place of their sets):
// Base.isambiguous's rule, at q. Both match q, neither is more specific than the other, and no
// matching method more specific than both covers q.
static int ambiguous_at(jl_methtable_t *mt, jl_method_t *mX, jl_method_t *mY, jl_value_t *q, size_t world)
{
    if (jl_method_morespecific(mX, mY) || jl_method_morespecific(mY, mX))
        return 0;
    size_t min_valid = 1, max_valid = ~(size_t)0;
    int ambig = 0;
    jl_value_t *res = ml_matches(mt, jl_method_table->cache, (jl_tupletype_t*)q, -1, 1, 1,
                                 world, 0, &min_valid, &max_valid, &ambig);
    if (!jl_is_array(res))
        return 0;
    JL_GC_PUSH1(&res);
    int hasX = 0, hasY = 0, dominated = 0;
    size_t l = jl_array_nrows((jl_array_t*)res);
    for (size_t i = 0; i < l; i++) {
        jl_method_t *m = ((jl_method_match_t*)jl_array_ptr_ref((jl_array_t*)res, i))->method;
        if (m == mX)
            hasX = 1;
        else if (m == mY)
            hasY = 1;
        else if (jl_method_morespecific(m, mX) && jl_method_morespecific(m, mY) &&
                 jl_subtype(q, (jl_value_t*)m->sig))
            dominated = 1;
    }
    JL_GC_POP();
    return hasX && hasY && !dominated;
}

JL_DLLEXPORT void jl_membership_ambiguity_report(jl_value_t *member, jl_value_t *set_instance,
                                                 size_t entry_min_world)
{
    (void)set_instance; // published by the caller, so it is in the member's own TypeName list below
    if (!jl_atomic_load_relaxed(&allow_new_worlds))
        return;
    jl_value_t *member_core = jl_unwrap_unionall(member);
    if (!jl_is_datatype(member_core))
        return;
    jl_typename_t *mtn = ((jl_datatype_t*)member_core)->name;
    // gc slots: 0 sets, 1 listX, 2 listY, 3 and 4 the query's parts, 5 the functions warned
    // for the current pair of sets (values held directly)
    jl_value_t **gc;
    JL_GC_PUSHARGS(gc, 6);
    // sets = every set the member joins (§1's lists on the member's TypeName):
    // an entry (M, X) with member ⊆ M witnesses member ⊆ M ⊆ X. The svec stays
    // rooted through mtn across the allocations below (review sweep 7).
    jl_array_t *sets = (jl_array_t*)(gc[0] = (jl_value_t*)jl_alloc_vec_any(0));
    int member_ground = !jl_has_free_typevars(member);
    jl_svec_t *list = jl_typename_memberships(mtn);
    for (size_t i = 0; i < (size_t)jl_svec_len(list) && jl_array_nrows(sets) < 16; i++) {
        jl_membership_t *en = (jl_membership_t*)jl_svecref(list, i);
        if (en->min_world > entry_min_world)
            continue;
        int joins = member_ground ? jl_subtype(member, (jl_value_t*)en->member)
                                  : jl_egal(member, (jl_value_t*)en->member);
        if (!joins)
            continue;
        jl_value_t *Y = en->instance;
        // review W2-4 (=W3-1): whole-family/parametric instances are UnionAlls —
        // the raw ->name read handed out the typevar pointer as a bogus TypeName
        // key (pointer-compared, never dereferenced), so the pair was silently
        // dropped and a genuinely ambiguous parametric declaration never warned.
        jl_typename_t *yn = ((jl_datatype_t*)jl_unwrap_unionall(Y))->name;
        size_t k;
        for (k = 0; k < jl_array_nrows(sets); k++)
            if (((jl_datatype_t*)jl_unwrap_unionall(jl_array_ptr_ref(sets, k)))->name == yn)
                break;
        if (k == jl_array_nrows(sets))
            jl_array_ptr_1d_push(sets, Y);
    }
    size_t ns = jl_array_nrows(sets);
    int concrete = jl_is_concrete_type(member_core);
    for (size_t a = 0; a < ns; a++) { // pairs, once each
        for (size_t b = a + 1; b < ns; b++) {
            jl_value_t *X = jl_array_ptr_ref(sets, a);
            jl_value_t *Y = jl_array_ptr_ref(sets, b);
            jl_typename_t *tnX = ((jl_datatype_t*)jl_unwrap_unionall(X))->name; // W2-4: UnionAll instances
            jl_typename_t *tnY = ((jl_datatype_t*)jl_unwrap_unionall(Y))->name;
            if (tnX == tnY)
                continue;
            jl_array_t *listX = (jl_array_t*)(gc[1] = (jl_value_t*)jl_methods_at_set(tnX));
            jl_array_t *listY = (jl_array_t*)(gc[2] = (jl_value_t*)jl_methods_at_set(tnY));
            if (jl_array_nrows(listX) == 0 || jl_array_nrows(listY) == 0) {
                gc[1] = gc[2] = NULL;
                continue;
            }
            // one warning per function and pair of sets
            jl_array_t *warned = (jl_array_t*)(gc[5] = (jl_value_t*)jl_alloc_vec_any(0));
            for (size_t x = 0; x < jl_array_nrows(listX); x++) {
                jl_method_t *mX = (jl_method_t*)jl_array_ptr_ref(listX, x);
                jl_methtable_t *mtX = jl_method_get_table(mX);
                if (!jl_is_mtable(mtX))
                    continue;
                jl_value_t *mbX = jl_unwrap_unionall((jl_value_t*)mX->sig);
                if (!jl_is_tuple_type(mbX) || jl_nparams((jl_datatype_t*)mbX) == 0)
                    continue;
                jl_value_t *fX = jl_tparam0((jl_datatype_t*)mbX); // the function's type
                for (size_t y = 0; y < jl_array_nrows(listY); y++) {
                    jl_method_t *mY = (jl_method_t*)jl_array_ptr_ref(listY, y);
                    if (mY == mX || jl_method_get_table(mY) != mtX)
                        continue;
                    jl_value_t *mbY = jl_unwrap_unionall((jl_value_t*)mY->sig);
                    if (!jl_is_tuple_type(mbY) || jl_nparams((jl_datatype_t*)mbY) == 0 ||
                        !jl_egal(jl_tparam0((jl_datatype_t*)mbY), fX))
                        continue; // another function
                    int seen = 0;
                    for (size_t w = 0; w < jl_array_nrows(warned) && !seen; w++)
                        seen = jl_egal(jl_array_ptr_ref(warned, w), fX);
                    if (seen)
                        continue;
                    if (concrete) {
                        // the call types at which both methods apply to the member: each method's
                        // signature with the member in place of its set, intersected (compiler §5.4).
                        // Empty: no call reaches both. Else Base.isambiguous's rule at those types, in
                        // the entry's world (the counter is published to it before the report).
                        gc[3] = sig_at_member((jl_value_t*)mX->sig, tnX, member);
                        gc[4] = sig_at_member((jl_value_t*)mY->sig, tnY, member);
                        if (gc[3] != NULL && gc[4] != NULL) {
                            gc[3] = jl_type_intersection(gc[3], gc[4]);
                            if (gc[3] != jl_bottom_type &&
                                ambiguous_at(mtX, mX, mY, gc[3], entry_min_world)) {
                                // the joint method: the two signatures' intersection, where the
                                // two sets meet as All{X, Y} (§4.9); at the member where it is empty
                                gc[4] = jl_type_intersection((jl_value_t*)mX->sig, (jl_value_t*)mY->sig);
                                membership_ambig_warn(mX, X, Y, member, gc[4] != jl_bottom_type ? gc[4] : gc[3]);
                                jl_array_ptr_1d_push(warned, fX);
                            }
                        }
                        gc[3] = gc[4] = NULL;
                    }
                    else {
                        // abstract/parametric member: no instance to ask. Suppress
                        // when some indexed method is more specific than both sides.
                        // The candidates are the two sets' own lists AND the methods
                        // indexed under any open set Q that itself joins X or Y by
                        // the entry chain (M2a fix round 2, gate m2_ambiguity_report:
                        // the set-in-set method g(::ZB), ZB ∈ XB ∧ ZB ∈ YB, resolves
                        // the pair for every member of ZB — it is indexed under ZB,
                        // so the exact-name lists alone missed it and over-warned).
                        // A closed method on both sides not among these is missed —
                        // over-warn (deviations.md).
                        int suppressed = 0;
                        for (int l = 0; l < 2 && !suppressed; l++) {
                            jl_array_t *l3 = l == 0 ? listX : listY;
                            for (size_t z = 0; z < jl_array_nrows(l3) && !suppressed; z++) {
                                jl_method_t *m3 = (jl_method_t*)jl_array_ptr_ref(l3, z);
                                if (m3 == mX || m3 == mY)
                                    continue;
                                if (jl_type_morespecific((jl_value_t*)m3->sig, (jl_value_t*)mX->sig) &&
                                    jl_type_morespecific((jl_value_t*)m3->sig, (jl_value_t*)mY->sig))
                                    suppressed = 1;
                            }
                        }
                        if (!suppressed) {
                            JL_LOCK(&open_set_methods_lock);
                            size_t ml_ = open_set_methods ? jl_array_nrows(open_set_methods) : 0;
                            for (size_t i = 0; i + 1 < ml_ && !suppressed; i += 2) {
                                jl_typename_t *qn = (jl_typename_t*)jl_array_ptr_ref(open_set_methods, i);
                                if (qn == tnX || qn == tnY || !qn->wrapper)
                                    continue;
                                jl_method_t *m3 = (jl_method_t*)jl_array_ptr_ref(open_set_methods, i + 1);
                                if (m3 == mX || m3 == mY || jl_method_get_table(m3) != mtX)
                                    continue;
                                gc[3] = (jl_value_t*)m3; // keep the pair rooted across the tests below
                                if (!(jl_membership_holds_at((jl_value_t*)qn->wrapper, X, entry_min_world) ||
                                      jl_membership_holds_at((jl_value_t*)qn->wrapper, Y, entry_min_world)))
                                    continue;
                                if (jl_type_morespecific((jl_value_t*)m3->sig, (jl_value_t*)mX->sig) &&
                                    jl_type_morespecific((jl_value_t*)m3->sig, (jl_value_t*)mY->sig))
                                    suppressed = 1;
                            }
                            gc[3] = NULL;
                            JL_UNLOCK(&open_set_methods_lock);
                        }
                        if (!suppressed) {
                            membership_ambig_warn(mX, X, Y, member, NULL);
                            jl_array_ptr_1d_push(warned, fX);
                        }
                    }
                }
            }
            gc[1] = gc[2] = gc[5] = NULL;
        }
    }
    JL_GC_POP();
}

// see if it might be possible to construct an instance of `typ`
// if n_uninitialized == 0, but a fieldtype is Union{},
// that type will not be constructable, for example, tested recursively
JL_DLLEXPORT int jl_has_concrete_subtype(jl_value_t *typ)
{
    if (typ == jl_bottom_type)
        return 0;
    typ = jl_unwrap_unionall(typ);
    if (jl_is_vararg(typ))
        typ = jl_unwrap_vararg(typ);
    if (!jl_is_datatype(typ))
        return 1;
    return ((jl_datatype_t*)typ)->has_concrete_subtype;
}

JL_DLLEXPORT uint64_t jl_typeinf_timing_begin(void)
{
    jl_task_t *ct = jl_current_task;
    if (ct->reentrant_timing & 1)
        return 0;
    ct->reentrant_timing |= 1;
    return jl_hrtime();
}

JL_DLLEXPORT void jl_typeinf_timing_end(uint64_t start, int is_recompile)
{
    if (!start)
        return;
    jl_task_t *ct = jl_current_task;
    ct->reentrant_timing &= ~1u;
    if (jl_atomic_load_relaxed(&jl_measure_compile_time_enabled)) {
        uint64_t inftime = jl_hrtime() - start;
        jl_atomic_fetch_add_relaxed(&jl_cumulative_compile_time, inftime);
        if (is_recompile) {
            jl_atomic_fetch_add_relaxed(&jl_cumulative_recompile_time, inftime);
        }
    }
}

// declare a C-callable entry point; called during code loading from the toplevel
JL_DLLEXPORT void jl_extern_c(jl_value_t *name, jl_value_t *declrt, jl_tupletype_t *sigt)
{
    // validate arguments. try to do as many checks as possible here to avoid
    // throwing errors later during codegen.
    JL_TYPECHK(@ccallable, type, declrt);
    if (!jl_is_tuple_type(sigt))
        jl_type_error("@ccallable", (jl_value_t*)jl_anytuple_type_type, (jl_value_t*)sigt);
    // check that f is a guaranteed singleton type
    jl_datatype_t *ft = (jl_datatype_t*)jl_tparam0(sigt);
    if (!jl_is_datatype(ft) || !jl_is_datatype_singleton(ft))
        jl_error("@ccallable: function object must be a singleton");

    // compute / validate return type
    if (!jl_type_mappable_to_c(declrt))
        jl_error("@ccallable: return type doesn't correspond to a C type");

    // validate method signature
    size_t i, nargs = jl_nparams(sigt);
    for (i = 1; i < nargs; i++) {
        jl_value_t *ati = jl_tparam(sigt, i);
        if (!jl_is_concrete_type(ati) || jl_is_kind(ati) || !jl_type_mappable_to_c(ati))
            jl_error("@ccallable: argument types must be concrete");
    }

    // save a record of this so that the alias is generated when we write an object file
    jl_method_t *meth = (jl_method_t*)jl_methtable_lookup((jl_value_t*)sigt, jl_atomic_load_acquire(&jl_world_counter));
    if (!jl_is_method(meth))
        jl_error("@ccallable: could not find requested method");
    JL_GC_PUSH1(&meth);
    if (name == jl_nothing)
        meth->ccallable = jl_svec2(declrt, (jl_value_t*)sigt);
    else
        meth->ccallable = jl_svec3(declrt, (jl_value_t*)sigt, name);
    jl_gc_wb(meth, meth->ccallable);
    JL_GC_POP();
}

// Drop all method caches and increment world age as if adding a method that intersects everything
static void invalidate_method_instance_caches(jl_method_instance_t *mi, size_t world)
{
    if ((jl_value_t*)mi == jl_nothing)
        return;

    // Walk through all code instances for this method instance
    jl_code_instance_t *ci = jl_atomic_load_relaxed(&mi->cache);
    while (ci != NULL) {
        // Invalidate this code instance by setting max_world to current world
        if (jl_atomic_load_relaxed(&ci->max_world) == ~(size_t)0) {
            jl_atomic_store_release(&ci->max_world, world);
        }
        ci = jl_atomic_load_relaxed(&ci->next);
    }
}

static int invalidate_all_specializations(jl_typemap_entry_t *def, void *closure)
{
    size_t world = *(size_t*)closure;
    jl_method_t *method = def->func.method;
    JL_LOCK(&method->writelock);
    jl_value_t *specializations = jl_atomic_load_relaxed(&method->specializations);
    if (jl_is_svec(specializations)) {
        size_t i, l = jl_svec_len(specializations);
        for (i = 0; i < l; i++) {
            jl_method_instance_t *mi = (jl_method_instance_t*)jl_svecref(specializations, i);
            invalidate_method_instance_caches(mi, world);
        }
    }
    else if (specializations != NULL) {
        jl_method_instance_t *mi = (jl_method_instance_t*)specializations;
        invalidate_method_instance_caches(mi, world);
    }
    JL_UNLOCK(&method->writelock);
    return 1;
}

static int invalidate_all_caches_visitor(jl_methtable_t *mt, void *env)
{
    return jl_typemap_visitor(jl_atomic_load_relaxed(&mt->defs), invalidate_all_specializations, env);
}

JL_DLLEXPORT void jl_drop_all_caches(void)
{
    JL_LOCK(&world_counter_lock);

    // Get current world age - we'll invalidate everything at this world
    size_t current_world = jl_atomic_load_relaxed(&jl_world_counter);

    invalidate_all_caches_visitor(jl_method_table, &current_world);

    // Increment world age - this forces all subsequent compilation to happen in the new world
    size_t new_world = current_world + 1;
    jl_atomic_store_release(&jl_world_counter, new_world);

    JL_UNLOCK(&world_counter_lock);
}


#ifdef __cplusplus
}
#endif

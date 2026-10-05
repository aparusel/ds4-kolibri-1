/* Kolibri-1 session-continuation gate.
 *
 * A session sync that extends an existing checkpoint must produce the same
 * next-token logits as a fresh prefill of the full prompt. This covers the
 * CPU reference continuation path, which has its own per-token forward and
 * must not fall through to the DeepSeek decode code, and the chunked-prefill
 * continuation of the default backend.
 *
 * Usage: tests/test_kolibri1_session [--cpu] [model.gguf]
 */
#include "../ds4.c"

#include <math.h>

static ds4_backend test_default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void push_ids(ds4_tokens *t, const int *ids, int count) {
    for (int i = 0; i < count; i++) ds4_tokens_push(t, ids[i]);
}

int main(int argc, char **argv) {
    bool cpu = false;
    const char *model = "gguf/Kolibri-1-mini.gguf";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu")) cpu = true;
        else model = argv[i];
    }

    ds4_engine_options opt = {0};
    opt.model_path = model;
    opt.backend = cpu ? DS4_BACKEND_CPU : test_default_backend();
    opt.context_size = 512;
    opt.power_percent = 100;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "test_kolibri1_session: engine open failed for %s\n", model);
        return 2;
    }

    static const int ids[] = {11, 22, 33, 44, 55, 66};
    const int base_len = 3;
    const int full_len = 6;
    const int vocab = ds4_engine_vocab_size(e);
    float *live = malloc((size_t)vocab * sizeof(live[0]));
    float *fresh = malloc((size_t)vocab * sizeof(fresh[0]));
    ds4_tokens base = {0}, full = {0};
    ds4_session *cont = NULL, *ref = NULL;
    int rc = 1;
    char err[160];
    if (!live || !fresh) {
        fprintf(stderr, "test_kolibri1_session: out of memory\n");
        goto done;
    }
    push_ids(&base, ids, base_len);
    push_ids(&full, ids, full_len);

    if (ds4_session_create(&cont, e, 512) != 0 ||
        ds4_session_create(&ref, e, 512) != 0) {
        fprintf(stderr, "test_kolibri1_session: session create failed\n");
        goto done;
    }
    if (ds4_session_sync(cont, &base, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: base sync failed: %s\n", err);
        goto done;
    }
    if (ds4_session_sync(cont, &full, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: continuation sync failed: %s\n", err);
        goto done;
    }
    if (ds4_session_sync(ref, &full, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: fresh sync failed: %s\n", err);
        goto done;
    }
    if (ds4_session_copy_logits(cont, live, vocab) != vocab ||
        ds4_session_copy_logits(ref, fresh, vocab) != vocab) {
        fprintf(stderr, "test_kolibri1_session: logits copy failed\n");
        goto done;
    }

    float max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        const float d = fabsf(live[i] - fresh[i]);
        if (d > max_abs) max_abs = d;
    }
    const int cont_argmax = ds4_session_argmax(cont);
    const int ref_argmax = ds4_session_argmax(ref);
    const bool ok = cont_argmax == ref_argmax && max_abs <= 1.0e-5f;
    printf("kolibri1-session %s: continuation vs fresh max|d| %.3g "
           "argmax %d/%d %s\n",
           cpu ? "cpu" : "default", max_abs, cont_argmax, ref_argmax,
           ok ? "OK" : "FAIL");
    rc = ok ? 0 : 1;

done:
    if (cont) ds4_session_free(cont);
    if (ref) ds4_session_free(ref);
    ds4_tokens_free(&base);
    ds4_tokens_free(&full);
    free(live);
    free(fresh);
    ds4_engine_close(e);
    return rc;
}

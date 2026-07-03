import httpx
import time

BASE = "http://localhost:8000"
results = []

def check(name, payload, checks):
    r = httpx.post(BASE + "/chat", json={"messages": payload}, timeout=35)
    d = r.json()
    ok = True
    failures = []
    for k, v in checks.items():
        actual = d.get(k)
        if actual != v:
            ok = False
            failures.append(f"  expected {k}={repr(v)}, got {repr(actual)}")
    tag = "PASS" if ok else "FAIL"
    print("[" + tag + "] " + name)
    for f in failures:
        print(f)
    results.append((name, ok))
    return ok

if __name__ == "__main__":
    # 1. Vague opening -> clarify, recs=[], EOC=False
    check("vague_opening",
        [{"role": "user", "content": "I need an assessment."}],
        {"end_of_conversation": False, "recommendations": []})

    # 2. Out-of-scope -> recs=[], EOC=False
    check("out_of_scope",
        [{"role": "user", "content": "Am I legally required under HIPAA to test all staff?"}],
        {"end_of_conversation": False, "recommendations": []})

    # 3. Confirmation phrase -> EOC=True (via keyword boost)
    check("confirmation_eoc",
        [
            {"role": "user", "content": "Need assessments for a senior Java engineer."},
            {"role": "assistant", "content": "| 1 | Core Java (Advanced Level) (New) | K | https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/ |"},
            {"role": "user", "content": "Perfect, that covers it. Confirmed."}
        ],
        {"end_of_conversation": True})

    # 4. Health endpoint
    r = httpx.get(BASE + "/health")
    ok4 = r.status_code == 200 and r.json() == {"status": "ok"}
    tag = "PASS" if ok4 else "FAIL"
    print("[" + tag + "] health")
    results.append(("health", ok4))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("\n" + str(passed) + "/" + str(total) + " tests passed")

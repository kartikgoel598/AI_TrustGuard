import ast
import pandas as pd

from detection.rag_poison_detector.clean_base_detector import CleanBaseDetector

CLEAN_DOCS_PATH = "dataset/rag_dataset/clean_documents.csv"
CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
TEST_CASES_PATH = "dataset/rag_dataset/rag_test_cases.csv"


def main():
    clean_docs = pd.read_csv(CLEAN_DOCS_PATH)
    corpus = pd.read_csv(CORPUS_PATH)

    contents = dict(zip(clean_docs["document_id"], clean_docs["content"]))
    contents.update(dict(zip(corpus["document_id"], corpus["content"])))

    tests = pd.read_csv(TEST_CASES_PATH)
    poison_tests = tests[tests["test_type"].isin(["basic_poison", "adaptive_poison", "mixed_poison"])]

    detector = CleanBaseDetector(device="cuda")

    totals = {"scenarios": 0, "poison_total": 0, "poison_flagged": 0,
              "clean_total": 0, "clean_flagged": 0, "any_clique": 0}

    for test_type in poison_tests["test_type"].unique():
        subset = poison_tests[poison_tests["test_type"] == test_type]
        t = {"scenarios": 0, "poison_total": 0, "poison_flagged": 0,
             "clean_total": 0, "clean_flagged": 0, "any_clique": 0}

        for _, row in subset.iterrows():
            doc_ids = ast.literal_eval(row["corpus_document_ids"])
            poison_ids = set(ast.literal_eval(row["target_poison_document_ids"]))

            missing = [d for d in doc_ids if d not in contents]
            if missing:
                continue

            documents = [contents[d] for d in doc_ids]
            if len(documents) < 11:
                continue

            result = detector.run({"documents": documents})
            if result.status == "error":
                continue

            flagged_ids = {doc_ids[d["index"]] for d in result.detail["per_document"] if d["flagged"]}

            t["scenarios"] += 1
            t["any_clique"] += int(len(result.detail["cliques"]) > 0)
            for doc_id in doc_ids:
                if doc_id in poison_ids:
                    t["poison_total"] += 1
                    t["poison_flagged"] += int(doc_id in flagged_ids)
                else:
                    t["clean_total"] += 1
                    t["clean_flagged"] += int(doc_id in flagged_ids)

        print(f"{test_type}: scenarios={t['scenarios']} scenarios_with_a_clique={t['any_clique']} "
              f"poison_flagged={t['poison_flagged']}/{t['poison_total']} "
              f"clean_flagged={t['clean_flagged']}/{t['clean_total']}")

        for k in totals:
            totals[k] += t[k]

    print(f"\noverall: scenarios={totals['scenarios']} scenarios_with_a_clique={totals['any_clique']} "
          f"poison_flagged={totals['poison_flagged']}/{totals['poison_total']} "
          f"clean_flagged={totals['clean_flagged']}/{totals['clean_total']}")


if __name__ == "__main__":
    main()
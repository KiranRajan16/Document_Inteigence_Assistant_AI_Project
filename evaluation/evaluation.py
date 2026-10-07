import pandas as pd
import sys
from pathlib import Path

# Import files from src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root/ "src"))
from graph import graph

# Loading evaluation dataset
dataset_path = project_root / "evaluation" / "evaluation_data.csv"
df = pd.read_csv(dataset_path)

results = []

print("\n_______________________________")
print("RAG Evaluation Started")
print("_________________________________")

for _, row in df.iterrows():
    question = row["question"]

    print("\nQuestion {row['id']} : {question}")

    try:
        result = graph.invoke({
            "question":question,
            "context": "",
            "answer": "",
            "sources": [],
            "grounded": False,
            "retry_count":0
        })

        answer = result["answer"]
        grounded = result["grounded"]

        sources = ";".join(
            f"{source['source']} - Page{source['page']}"
            for source in result['sources']
        )

        results.append({
            "id": row["id"],
            "question": question,
            "type": row['type'],
            "answer": answer,
            "grounded": grounded,
            "sources": sources
        })

        print("completed")

    except Exception as e:
        print(f"Error: {e}")

        results.append({
            "id": row["id"],
            "question": question,
            "type": row["type"],
            "answer": "ERROR",
            "grounded": False,
            "sources": ""
        })

# create results dataframe
results_df = pd.DataFrame(results)
# Save results
output_path = project_root/"evaluation"/"evaluation_results.csv"

results_df.to_csv(output_path, index=False)

print("\n_______________________________")
print("Evaluation Completed ")
print("_________________________________")

print(f"Total Questions: {len(results_df)}")
print(f"Results saved to: {output_path}")
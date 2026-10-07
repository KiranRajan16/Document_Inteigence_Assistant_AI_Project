import pandas as pd
from pathlib import Path

# project root
project_root = Path(__file__).resolve().parent.parent
# evaluation results file
results_path = project_root/ "evaluation" / "evaluation_results.csv"
# load results
df = pd.read_csv(results_path)

print("\n_______________________________")
print("RAG Evaluation Metrics")
print("_________________________________")

total_questions = len(df)

grounded_count = df["grounded"].astype(str).str.lower().eq("true").sum()

ungrounded_count = total_questions - grounded_count

grounded_rate = (
    grounded_count/total_questions * 100
    if total_questions > 0
    else 0
)

print(f"Total Questions: {total_questions}")
print(f"Grounded Answers: {grounded_count}")
print(f"Ungrounded Answers: {ungrounded_count}")
print(f"Grounded Rate: {grounded_rate:.2f}%")

print("\n_________________________________")
print("Results by Question Type")
print("___________________________________")

type_results =(
    df.groupby("type")["grounded"]
    .apply(lambda x : x.astype(str).str.lower().eq("true").sum())
)

for question_type, count in type_results.items():
    total = len(df[df["type"] == question_type])
    percentage = count/ total * 100 if total > 0 else 0 

    print(
        f"{question_type:20}: "
        f"{count}/{total} grounded ({percentage:.2f}%)"
    )

print("\n____________________________")
print("Metrics Completed")
print("______________________________")
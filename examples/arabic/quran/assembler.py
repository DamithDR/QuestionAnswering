import json
import os
import sys


def get_score(ans):
    return ans.get('score')


def assemble_results(files=[], output_file_name="DTW_01", ans_limit=None):
    results_file = os.path.join("examples", "arabic", "quran", "data", "run-files", output_file_name + ".json")

    jsons = []
    for file in files:
        f = open(file, 'r', encoding='utf-8')
        jsons.append(json.load(f))

    keys = None
    if len(jsons) == 0:
        print('No jsons loaded')
        sys.exit(0)
    else:
        keys = jsons[0].keys()

    print('Total No of Keys :' + str(len(keys)))

    ans_scores_assembled_dict = {}

    for key in keys:
        temp_arr = []
        for jsn in jsons:
            temp_arr.append(jsn[key])
        assembeled_answers = []

        unique_answers = set()
        for t in temp_arr:
            for ans in t:
                unique_answers.add(ans['answer'])

        for unique_ans in unique_answers:
            ans_score = 0.0
            for t in temp_arr:
                for ans in t:
                    if unique_ans == ans['answer']:
                        ans_score = ans_score + ans['score']
            assembeled_answers.append({'answer': unique_ans, 'score': (ans_score / len(jsons) * 1.0), 'rank': 0})

        # sort and re-rank
        # print("unsorted ", assembeled_answers)
        assembeled_answers.sort(key=get_score, reverse=True)
        i = 1
        for asem_ans in assembeled_answers:
            asem_ans['rank'] = i
            i += 1
        # print("sorted", assembeled_answers)

        if ans_limit is None:
            ans_scores_assembled_dict[key] = assembeled_answers
        else:
            ans_scores_assembled_dict[key] = assembeled_answers[:ans_limit]

    with open(results_file, "w", encoding="utf-8") as outfile:
        json.dump(ans_scores_assembled_dict, outfile, ensure_ascii=False)
    return results_file


if __name__ == '__main__':
    assemble_results()

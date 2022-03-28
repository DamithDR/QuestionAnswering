import json
import os
import sys


def get_score(ans):
    return ans.get('score')


def assemble_results():
    path = os.path.join(".", "data", "run-files", "assemble")
    results_file = os.path.join(".", "data", "run-files", "DTW_01.json")

    files = os.listdir(path)
    jsons = []
    for file in files:
        f = open(os.path.join(path, file), 'r', encoding='utf-8')
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
                        if ans_score == 0:
                            ans_score = ans['score']
                        else:
                            ans_score = (ans_score + ans['score']) / 2.0
            assembeled_answers.append({'answer': unique_ans, 'score': ans_score, 'rank': 0})

        # sort and re-rank
        # print("unsorted ", assembeled_answers)
        assembeled_answers.sort(key=get_score, reverse=True)
        i = 1
        for asem_ans in assembeled_answers:
            asem_ans['rank'] = i
            i += 1
        # print("sorted", assembeled_answers)
        ans_scores_assembled_dict[key] = assembeled_answers[:4]

    with open(results_file, "w", encoding="utf-8") as outfile:
        json.dump(ans_scores_assembled_dict, outfile, ensure_ascii=False)


if __name__ == '__main__':
    assemble_results()

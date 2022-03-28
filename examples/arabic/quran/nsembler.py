import json
import os
import sys


def get_score(ans):
    return ans.get('score')


def ensemble():
    path = os.path.join(".", "data", "run-files", "assemble")

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
        for i in range(0, len(temp_arr) - 1):
            considering_list = temp_arr[i]
            for j in range(i + 1, len(temp_arr)):
                comparing_list = temp_arr[j].copy()
                for ans in considering_list:
                    answer = ans['answer']
                    score = ans['score']
                    # for compare in comparing_list:
                    ans_appended = False
                    for compare in comparing_list:
                        if answer == compare['answer']:
                            score = (ans['score'] + compare['score']) / 2.0
                            comparing_list.remove(compare)
                            assembeled_answers.append({'answer': ans['answer'], 'score': score, 'rank': 0})
                            ans_appended = True
                        else:
                            if not ans_appended:
                                assembeled_answers.append({'answer': ans['answer'], 'score': ans['score'], 'rank': 0})
                                ans_appended = True

                temp_arr[j] = comparing_list

        if len(temp_arr[len(temp_arr) - 1]) > 0:
            for left_ans in temp_arr[len(temp_arr) - 1]:
                assembeled_answers.append({'answer': left_ans['answer'], 'score': left_ans['score'], 'rank': 0})

        # sort and re-rank
        print("unsorted ", assembeled_answers)
        assembeled_answers.sort(key=get_score, reverse=True)
        i = 1
        for asem_ans in assembeled_answers:
            asem_ans['rank'] = i
            i += 1
        print("sorted" ,assembeled_answers)


if __name__ == '__main__':
    ensemble()

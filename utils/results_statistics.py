import numpy as np

def write_file(output_path: str, data: np.ndarray, encoding: str = 'utf-8') -> bool:
    result = ''
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    for i in range(data.shape[0]):
        line = ''
        for j in data[i]:
            line += str(j) + '\t'
        result += line[:-1] + '\n'
    with open(output_path, 'w', encoding=encoding) as file:
        file.write(result)
    return True

def divide_results(input_path: str, separator: str = '\t', column: int = 2, encoding: str = 'utf-8') -> np.ndarray:
    result = []
    with open(input_path, 'r', encoding=encoding) as file:
        for line in file:
            line = line.rstrip('\n') 
            parts = line.split(separator)

            if len(parts) < column:
                parts.extend([''] * (column - len(parts)))

            if len(parts) > column:
                for i in range(len(parts) - column):
                    parts[column - 1] += parts[column + i]

            result.append(parts[:column])
    return np.array(result)

def is_in_osr_char(osr_char: np.ndarray, char_gt: str) -> bool:
    for char in char_gt:
        if np.all(np.char.find(osr_char, char) == -1):
            return False
    return True


# Yulin Huang: 单行数据写入文件记录
def write_line(path: str, data: list):
    with open(path, 'a') as file:
        line = ''
        for d in data:
            line += str(d) + '\t'
        file.write(line[:-1] + '\n')


if __name__ == '__main__':

    # osr_char('./swin-tr/rec_gt_train.txt', '.')
    # train_char = divide_results('char_osr_rec_gt_train.txt')[..., 0]
    # gt = divide_results('./swin-tr/rec_gt_test.txt')[..., 1]
    # result = []
    # for char in gt:
    #     result.append(is_in_osr_char(train_char, char))
    # write_file('test.txt', np.array(result))

    # data_train = divide_results('data_train.txt', column=5)
    # write_file('data_train.txt', data_train[:218])

    # gt = divide_results('./swin-tr/rec_gt_test.txt')
    # pred = divide_results('./test_fusion_pred.txt')
    # write_back_path = 'results_statistics.txt'
    # compare_pred(pred, gt, write_back_path)

    pass
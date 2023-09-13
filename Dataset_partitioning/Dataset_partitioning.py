import json
import random


# json >> list
def dataset_part(dataset, seed_num=30):
    # 随机种子,确保可重复性
    random.seed(seed_num)

    # 数据集大小 列表
    total_size = len(dataset)

    # 划分比例，例如 70%训练集，15% 验证集，15% 测试集
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 计算划分后的数据集大小
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)

    # 创建索引列表并随机打乱
    indices = list(range(total_size))
    random.shuffle(indices)

    # 划分数据集
    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):]

    # 根据索引获取数据
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    with open(r"/bert/data/data_pad/train.json", "r", encoding='utf-8') as file:
        str = json.loads(file.read())
        print(len(dataset_part(str)[2]))
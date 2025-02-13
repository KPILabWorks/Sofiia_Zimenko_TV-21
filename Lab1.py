from typing import List


def find_sublists(lst: List[int], length: int) -> List[List[int]]:
    """
    Функція знаходить усі можливі підсписки заданої довжини у списку.
    :param lst: Вхідний список.
    :param length: Довжина підсписку.
    :return: Список підсписків заданої довжини або повідомлення про помилку.
    """
    if length <= 0:
        print("Помилка: Довжина підсписку має бути більше 0.")
        return []
    if length > len(lst):
        print("Помилка: Довжина підсписку перевищує довжину списку.")
        return []

    return [lst[i:i + length] for i in range(len(lst) - length + 1)]


try:
    example_list = list(map(int, input("Введіть список чисел через пробіл: ").split()))
    sublist_length = int(input("Введіть довжину підсписку: "))
    result = find_sublists(example_list, sublist_length)
    if result:
        print("Знайдені підсписки:", result)
except ValueError:
    print("Помилка: Введіть коректні числові значення.")

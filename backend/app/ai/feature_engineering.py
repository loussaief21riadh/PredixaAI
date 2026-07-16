from statistics import mean, median


class FeatureEngineering:
    """
    Construit les variables (features) utilisées par les modèles IA.
    """

    @staticmethod
    def build(numbers: list[int]) -> dict:

        numbers = sorted(numbers)

        even = sum(1 for n in numbers if n % 2 == 0)
        odd = 5 - even

        consecutive = 0

        for i in range(4):
            if numbers[i + 1] == numbers[i] + 1:
                consecutive += 1

        decades = {
            "decade_1_9": 0,
            "decade_10_19": 0,
            "decade_20_29": 0,
            "decade_30_39": 0,
            "decade_40_49": 0,
        }

        for n in numbers:

            if 1 <= n <= 9:
                decades["decade_1_9"] += 1

            elif 10 <= n <= 19:
                decades["decade_10_19"] += 1

            elif 20 <= n <= 29:
                decades["decade_20_29"] += 1

            elif 30 <= n <= 39:
                decades["decade_30_39"] += 1

            elif 40 <= n <= 49:
                decades["decade_40_49"] += 1

        features = {

            "n1": numbers[0],
            "n2": numbers[1],
            "n3": numbers[2],
            "n4": numbers[3],
            "n5": numbers[4],

            "sum": sum(numbers),

            "mean": mean(numbers),

            "median": median(numbers),

            "min": min(numbers),

            "max": max(numbers),

            "range": max(numbers) - min(numbers),

            "even_count": even,

            "odd_count": odd,

            "consecutive_pairs": consecutive,

            **decades,
        }

        return features
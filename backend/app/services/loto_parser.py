from datetime import datetime


class LotoParser:
    """
    Convertit une ligne CSV FDJ en format Predixa AI.
    """

    @staticmethod
    def parse(row: dict):

        return {
            "draw_date": datetime.strptime(
                row["date_de_tirage"],
                "%Y%m%d",
            ).date(),

            "n1": int(row["boule_1"]),
            "n2": int(row["boule_2"]),
            "n3": int(row["boule_3"]),
            "n4": int(row["boule_4"]),
            "n5": int(row["boule_5"]),
            "n6": int(row["boule_6"]),

            "bonus": (
                int(row["boule_complementaire"])
                if row["boule_complementaire"]
                else None
            ),
        }
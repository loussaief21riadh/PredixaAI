from app.statistics.frequency import FrequencyAnalyzer
from app.statistics.hot_numbers import HotNumbersAnalyzer
from app.statistics.cold_numbers import ColdNumbersAnalyzer
from app.statistics.overdue import OverdueAnalyzer
from app.statistics.pair_analyzer import PairAnalyzer
from app.statistics.triplet_analyzer import TripletAnalyzer
from app.statistics.even_odd import EvenOddAnalyzer
from app.statistics.sum_analyzer import SumAnalyzer
from app.statistics.decade_analyzer import DecadeAnalyzer
from app.statistics.consecutive_analyzer import ConsecutiveAnalyzer


class DashboardAnalyzer:

    @staticmethod
    def calculate(db):

        return {
            "frequency": FrequencyAnalyzer.calculate(db),
            "hot": HotNumbersAnalyzer.calculate(db),
            "cold": ColdNumbersAnalyzer.calculate(db),
            "overdue": OverdueAnalyzer.calculate(db),
            "pairs": PairAnalyzer.calculate(db),
            "triplets": TripletAnalyzer.calculate(db),
            "even_odd": EvenOddAnalyzer.calculate(db),
            "sums": SumAnalyzer.calculate(db),
            "decades": DecadeAnalyzer.calculate(db),
            "consecutive": ConsecutiveAnalyzer.calculate(db),
        }
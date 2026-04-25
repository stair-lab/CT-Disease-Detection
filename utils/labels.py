from enum import IntEnum

class Condition(IntEnum):
    ABSENT = 0
    PRESENT = 1

    @staticmethod
    def convert(s):
        value = str(s).upper()
        if value == 'ABSENT':
            return Condition.ABSENT
        if value == 'PRESENT':
            return Condition.PRESENT
        raise ValueError(f"Unsupported condition label: {s}")


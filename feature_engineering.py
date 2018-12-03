# -*- encoding: UTF-8 -*-
from features.basic import TimeInformation
from features.confidence_rate import ConfidenceRate

if __name__ == "__main__":
    TimeInformation().run().save()
    ConfidenceRate().run().save()

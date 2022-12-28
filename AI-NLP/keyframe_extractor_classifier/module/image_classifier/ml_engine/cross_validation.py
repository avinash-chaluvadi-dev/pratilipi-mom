from sklearn import model_selection

from ..utils import custom_logging


class CrossValidation:
    def __init__(self, dataframe, target_columns, num_folds, shuffle=True):
        self.shuffle = shuffle
        self.dataframe = dataframe
        self.num_folds = num_folds
        self.target_columns = target_columns
        self.logger = custom_logging.get_logger()

    def split(self):
        if len(self.target_columns) == 1:
            unique_labels = self.dataframe.loc[:, self.target_columns[0]].nunique()
            if unique_labels == 1:
                self.logger.exception("Only one unique label found")
                raise
            elif unique_labels > 1:
                skf = model_selection.StratifiedKFold(
                    n_splits=self.num_folds,
                    shuffle=self.shuffle,
                    random_state=22,
                )
                for fold, (train_idx, validation_idx) in enumerate(
                    skf.split(
                        X=self.dataframe,
                        y=self.dataframe.loc[:, self.target_columns[0]].values,
                    )
                ):
                    self.dataframe.loc[validation_idx, "kf"] = fold
            return self.dataframe

        elif len(self.target_columns) > 1:
            self.logger.exception("Multi Label Classification Problem")
            raise
        else:
            self.logger.exception("Problem Type Not Understood")
            raise

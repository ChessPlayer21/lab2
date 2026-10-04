"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    matrix_data = pd.read_csv(data.csv_path)
    row_count = len(matrix_data)
    gaps_in_PassengerId = matrix_data["PassengerId"].isna().sum()
    gaps_in_Survived = matrix_data["Survived"].isna().sum()
    gaps_in_Pclass = matrix_data["Pclass"].isna().sum()
    gaps_in_Name = matrix_data["Name"].isna().sum()
    gaps_in_Sex = matrix_data["Sex"].isna().sum()
    gaps_in_Age = matrix_data["Age"].isna().sum()
    gaps_in_SibSp = matrix_data["SibSp"].isna().sum()
    gaps_in_Parch = matrix_data["Parch"].isna().sum()
    gaps_in_Ticket = matrix_data["Ticket"].isna().sum()
    gaps_in_Fare = matrix_data["Fare"].isna().sum()
    gaps_in_Cabin = matrix_data["Cabin"].isna().sum()
    gaps_in_Embarked = matrix_data["Embarked"].isna().sum()
    gaps = {
    "PassengerId": gaps_in_PassengerId,
    "Survived": gaps_in_Survived,
    "Pclass": gaps_in_Pclass,
    "Name": gaps_in_Name,
    "Sex": gaps_in_Sex,
    "Age": gaps_in_Age ,
    "SibSp": gaps_in_SibSp,
    "Parch": gaps_in_Parch,
    "Ticket": gaps_in_Ticket,
    "Fare": gaps_in_Fare,
    "Cabin": gaps_in_Cabin,
    "Embarked": gaps_in_Embarked,
}
    age_over_30 = (matrix_data["Age"] > 30).sum()
    age = matrix_data.groupby("Pclass")["Age"].mean().to_dict()
    age = {int(k): float(v) for k, v in age.items()}
    survived = matrix_data.groupby("Pclass")["Survived"].mean()
    survived = {int(k): float(v) for k, v in survived.items()}
    max = matrix_data["Fare"].nlargest(5)
    return TitanicSummary(
        row_count=row_count,
        missing_by_column=gaps,
        adults_over_30_count=age_over_30,
        mean_age_by_pclass=age,
        survival_rate_by_pclass=survived,
        highest_fares=max,
    )

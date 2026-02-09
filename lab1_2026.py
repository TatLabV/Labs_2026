import pandas as pd

def get_age_group(age):
    if age < 18:
        return "Ребенок"
    elif age <= 65:
        return "Взрослый"
    else:
        return "Пожилой"

def main():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    #Часть 1.1–1.3
    print("\n[1] Первые 7 строк:")
    print(df.head(7).to_string(index=False))

    print("\n[Ответ] Столбцы с пропусками:")
    missing_cols = df.columns[df.isna().any()].tolist()
    print(missing_cols)

    print("\n[1] info():")
    df.info()

    print("\n[1] describe():")
    print(df.describe().to_string())

    print("\n[Ответ] Средний возраст (Age mean):")
    print(f"{df['Age'].mean():.2f}")

    print("\n[Ответ] Максимальная цена билета (Fare max):")
    print(f"{df['Fare'].max():.2f}")

    #Часть 2.1
    df["Age"] = df["Age"].fillna(df["Age"].median())

    print("\n[Ответ] Количество пропусков в Age после заполнения:")
    print(df["Age"].isna().sum())

    #Часть 2.2
    df["AgeGroup"] = df["Age"].apply(get_age_group)

    print("\n[2] Проверка AgeGroup (первые 5 строк):")
    print(df[["Name", "Age", "AgeGroup"]].head(5).to_string(index=False))

    #Часть 3.1
    print("\n[Ответ] % выживших по полу:")
    surv_by_sex = (df.groupby("Sex")["Survived"].mean() * 100).round(2)
    print(surv_by_sex.to_string())

    print("\n[Ответ] % выживших по классу каюты:")
    surv_by_class = (df.groupby("Pclass")["Survived"].mean() * 100).round(2)
    print(surv_by_class.to_string())

    print("\n[Ответ] % выживших по полу и классу (сводная таблица):")
    pivot = (df.groupby(["Sex", "Pclass"])["Survived"].mean() * 100).round(2).unstack()
    print(pivot.to_string())

    #Часть 3.2
    print("\n[Ответ] Несовершеннолетние (<18), 3 класс, выжили (сортировка по возрасту убыв.):")
    q1 = df[(df["Age"] < 18) & (df["Pclass"] == 3) & (df["Survived"] == 1)][["Name","Age","Pclass","Survived"]]
    q1 = q1.sort_values("Age", ascending=False)
    print(q1.to_string(index=False))

    print("\n[Ответ] Самый пожилой мужчина, который не выжил:")
    q2 = df[(df["Sex"] == "male") & (df["Survived"] == 0)].sort_values("Age", ascending=False).head(1)
    print(q2[["Name","Age","Sex","Pclass","Fare","Survived"]].to_string(index=False))

if __name__ == "__main__":
    main()

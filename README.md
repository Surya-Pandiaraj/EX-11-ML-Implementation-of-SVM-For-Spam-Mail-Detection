### NAME: SURYA P <br>
### REG NO: 212224230280

# EX 11 : IMPLEMENTATION OF SVM FOR SPAM MAIL DETECTION

## AIM :
To write a program to implement the SVM For Spam Mail Detection.

## EQUIPMENTS REQUIRED :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM :

1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## PROGRAM :

```
Program to implement the SVM For Spam Mail Detection.
Developed by: Surya P
RegisterNumber: 212224230280
```

```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## OUTPUT :

![image](https://github.com/user-attachments/assets/ba42ca2d-85a5-4795-b40d-63df37236a81)


![image](https://github.com/user-attachments/assets/22edcf48-dd25-4513-867d-6c7430da985d)


![image](https://github.com/user-attachments/assets/618324db-35cd-4c45-8469-466a22c5cf86)


![image](https://github.com/user-attachments/assets/81bc5619-eb01-41f1-9218-e95a64875d42)


![image](https://github.com/user-attachments/assets/6836f245-d141-4e8b-8bfa-e2d8eac18a76)


![image](https://github.com/user-attachments/assets/3da73419-4d11-45bf-b361-b1fd344e732e)








## RESULT :
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

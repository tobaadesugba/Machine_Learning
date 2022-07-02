import pandas as pd

def TruthOrDare():
    questions = pd.read_excel("questions.xlsx", engine='', sheet_name="sheet1")
    print(questions.head())
    used_dares = []
    used_truths = []
    new_num = ''


def checkDare():
    while((num in TruthOrDare.used_dares) == False):
        num = input()
        TruthOrDare.new_num = num
        if num in TruthOrDare.used_dares:
            print("THIS NUMBER HAS ALREADY BEEN CHOSEN!")
    TruthOrDare.used_dares.append(num)

def checkTruth():
    while((num in TruthOrDare.used_truths) == False):
        num = input()
        TruthOrDare.new_num = num
        if num in TruthOrDare.used_truths:
            print("THIS NUMBER HAS ALREADY BEEN CHOSEN!")
    TruthOrDare.used_truths.append(num)


while(x == "again"):
    print("TRUTH OR DARE???")
    TorD = input()
    print("PICK A NUMBER BETWEEN 1 AND 40")
    if TorD == "truth":
        checkTruth()
    if TorD == "dare":
        checkDare()
    print(TruthOrDare.questions[TorD][int(TruthOrDare.new_num)])
    print()
    print("again???")
    x = input()

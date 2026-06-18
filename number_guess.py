answer = 7
user_guess = 7

print("숫자 맞히기 프로그램")

if user_guess == answer:
    print("정답입니다.")
elif user_guess > answer:
    print("입력한 숫자가 너무 큽니다.")
else:
    print("입력한 숫자가 너무 작습니다.")

import numpy as np
from matplotlib import pyplot as plt
import json
teacher_file='result.json'
student_file='result_student.json'
student_kd_file='result_kd.json'
def read_json(file):
    with open(file, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        print(json_data)
    return json_data

teacher_data=read_json(teacher_file)
student_data=read_json(student_file)
student_kd_data=read_json(student_kd_file)


x =[int(x) for x in  list(dict(teacher_data).keys())]
print(x)

plt.plot(x, list(teacher_data.values()), label='teacher')
plt.plot(x,list(student_data.values()), label='student without RKD')
plt.plot(x, list(student_kd_data.values()), label='student with RKD')

plt.title('Test accuracy')
plt.legend()


plt.show()
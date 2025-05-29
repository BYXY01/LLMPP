def return_hello_world(arg):
    # arg=eval(arg)
    # arg+=1
    return "hello world!"+str(arg)

dic={'call_function': 'return_hello_world', 'arg': '123'}
call_function=dic['call_function']
arg=dic['arg']
print(eval(f'{call_function}("{arg}")'))
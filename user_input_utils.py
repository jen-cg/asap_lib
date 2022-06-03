import sys
"""
User input utilities 

"""


# -----------------------------------------------------------------------------------------------------------------------
# Yes or no question with Yes and return as default
def yes_or_no(question):
    err = 1
    while err <= 3:
        reply = str(input(question)).lower()
        if 'y' in reply:
            return True
        if not reply:
            return True
        if 'n' in reply:
            return False
        else:
            err += 1
            print("Sorry, didn't catch that... ")
    if err >= 3:
        sys.exit('Three strikes and you are out: Did not understand input.')


# -----------------------------------------------------------------------------------------------------------------------
# No or yes question with No and return as default
def no_or_yes(question):
    err = 1
    while err <= 3:
        reply = str(input(question)).lower()
        if 'y' in reply:
            return False
        if not reply:
            return True
        if 'n' in reply:
            return True
        else:
            err = err + 1
            print("Sorry, didn't catch that... ")
    if err >= 3:
        sys.exit('Three strikes and you are out: Did not understand input')


# -----------------------------------------------------------------------------------------------------------------------
# Abundance type question. [X/Fe] or A(X) format? [X/Fe] is default
def fe_or_abs(question):
    err = 1
    while err <= 3:
        reply = str(input(question)).lower()
        if 'f' in reply:
            return True
        if not reply:
            return True
        if 'a' in reply:
            return False
        else:
            err = err + 1
            print("Sorry, didn't catch that... ")
    if err >= 3:
        sys.exit('Three strikes and you are out: Did not understand abundance type')


# -----------------------------------------------------------------------------------------------------------------------
# Interactive Parameter Check
def int_par_check(question, to_list):
    again = True
    err = 1
    while again:
        val = input(question)
        try:
            to_list.append(float(val))
            again = False
            return to_list
        except:
            print('Sorry, I do not recognize your input, try again ')
            err = err + 1
        if err > 3:
            again = False
            sys.exit('Three strikes and you are out! Careful with your input.')

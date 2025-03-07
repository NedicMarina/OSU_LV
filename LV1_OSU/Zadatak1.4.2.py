try:
    num = float(input("Unesite broj izmedu 0.0 i 1.0: "))
    if num < 0.0 or num > 1.0:
        print("Broj je izvan intervala [0.0, 1.0]")
    elif num >= 0.9:
        print("A")
    elif num >= 0.8:
        print("B")
    elif num >= 0.7:
        print("C")
    elif num >= 0.6:
        print("D")
    elif num < 0.6:
        print("F")    
except:
    print("Greska: Niste unijeli broj!")

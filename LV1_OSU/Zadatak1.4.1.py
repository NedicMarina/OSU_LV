

def total_euro(workingHours,salary_per_hour):
    return workingHours*salaryPerHour

workingHours=float(input("Broj radnih sati: "))
salaryPerHour=float(input("Plaća po satu: "))

print(f"Radni sati: {workingHours}")
print(f"eura/h: {salaryPerHour}")
print(f"Ukupno: {total_euro(workingHours, salaryPerHour)} eura")

def Average(lst): 
    return sum(lst) / len(lst)

file = open ('SMSSpamCollection.txt')

ham_word_counter = []
spam_word_counter = []
spam_ending_with_exclamation = 0

for line in file:
    line = line.rstrip ()
    words = line.split ()
    if words[0] == "ham":
        ham_word_counter.append(len(words)-1) #bez pocetnog ham
    elif words[0] == "spam":
        if words[-1].endswith("!"):
            spam_ending_with_exclamation += 1
        spam_word_counter.append(len(words)-1) #bez pocetnog spam
    else:
        print("Niti spam niti ham:") 
        print(line)

print(f"Prosječan broj riječi u ham porukama je {round(Average(ham_word_counter),2)}")
print(f"Prosječan broj riječi u spam porukama je {round(Average(spam_word_counter),2)}")
print(f"{spam_ending_with_exclamation} spam poruka završava uskličnikom")

file.close ()

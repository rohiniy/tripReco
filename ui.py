#!/usr/bin/python

import Tkinter
from user_based_collaborative_filtering import hybridRecommendation

master = Tkinter.Tk()

def show_entry_fields():
   print("City similar to: %s\n for User Id: %s" % (e1.get(), e2.get()))
   hybridRecommendations = hybridRecommendation(e2.get(), e1.get())
   print(hybridRecommendations)
   label1 = Tkinter.Label(master, text= "City similar to: %s\n for User Id: %s" % (e1.get(), e2.get()))
   label1.pack()

   label2 = Tkinter.Label(master, text= hybridRecommendations)
   label2.pack()

master.geometry('500x500')
frame = Tkinter.Frame(master, bg='grey')
frame.pack_propagate(0)
frame.pack(fill=Tkinter.BOTH, expand=1)
Tkinter.Label(master, text="Search City like").grid(row=0)
Tkinter.Label(master, text="User Id").grid(row=1)

e1 = Tkinter.Entry(master)
e2 = Tkinter.Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

Tkinter.Button(master, text='Show', command=show_entry_fields).grid(row=3, column=1, sticky=Tkinter.W, pady=4)

Tkinter.mainloop( )
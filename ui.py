#!/usr/bin/python

import Tkinter

master = Tkinter.Tk()

Tkinter.Label(master, text="Search City like").grid(row=0)
Tkinter.Label(master, text="Last Name").grid(row=1)

e1 = Tkinter.Entry(master)
e2 = Tkinter.Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

Tkinter.mainloop( )
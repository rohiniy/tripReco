#!/usr/bin/python

import Tkinter
from hybrid_filtering import hybridRecommendation
from content_based_filetering import get_city_recommendations
from simple_item_recommender import simple_recommender
from matrix_factorization_recommender import getRecommendedCities

master = Tkinter.Tk()
master.geometry('1028x1028')
master.title('Trip Recommender')

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox('all'))

def show_entry_fields():
   userId = e2.get()
   city = e1.get()
   labelContent1 = Tkinter.Label(frame,
                                text="City similar to: %s based on type of activities" % city).grid(row=5, column=1)
   labelContent2 = Tkinter.Label(frame,
                                 text="For a new system without any user ratings").grid(row=6, column=1)
   contentRecommendations = get_city_recommendations(city)
   labelContent3 = Tkinter.Label(frame, text=contentRecommendations['City'].head(10)).grid(row=7, column=1)

   Tkinter.Label(frame, text='--------------------------------------------------------------').grid(row=8, column=1)
   labelSimple1 = Tkinter.Label(frame,
                                 text="City similar to: %s based on all users ratings" % city).grid(row=9, column=1)
   labelSimple2 = Tkinter.Label(frame,
                                 text="For a new user without any ratings in a system with other users (not personalized)").grid(row=10, column=1)

   simpleRecommendations = simple_recommender(city)
   labelSimple3 = Tkinter.Label(frame, text=simpleRecommendations.head(10)).grid(row=11, column=1)

   Tkinter.Label(frame, text='--------------------------------------------------------------').grid(row=12, column=1)
   labelCollaborative1 = Tkinter.Label(frame,
                                text="New cities for User Id: %s according to user's past ratings based on other similar users"
                                     % userId).grid(row=13, column=1)
   labelCollaborative2 = Tkinter.Label(frame,
                                 text="New recommendations for an existing system and existing user (personalized content)").grid(row=14, column=1)

   collaborativeRecommendations = getRecommendedCities(int(userId))
   labelCollaborative3 = Tkinter.Label(frame, text=collaborativeRecommendations.head(10)).grid(row=15, column=1)

   Tkinter.Label(frame, text='--------------------------------------------------------------').grid(row=16, column=1)
   labelHybrid1 = Tkinter.Label(frame,
                                text="City similar to: %s for User Id: %s according to user's past ratings and type of activity in the city"
                                     % (city, userId)).grid(row=17, column=1)
   labelHybrid2 = Tkinter.Label(frame,
                                text="For an existing system and existing user (personalized content)").grid(row=18,
                                                                                                             column=1)

   hybridRecommendations = hybridRecommendation(int(userId), city)
   labelHybrid3 = Tkinter.Label(frame, text=hybridRecommendations.head(10)).grid(row=19, column=1)


canvas = Tkinter.Canvas(master, width=1028, height=1028, scrollregion=(0,0,1028,1028))

scrollbar = Tkinter.Scrollbar(master, command=canvas.yview)
scrollbar.pack(side=Tkinter.RIGHT, fill='y')

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', on_configure)

frame = Tkinter.Frame(canvas)
canvas.create_window((0,0), window=frame, anchor='nw')

Tkinter.Label(frame, text="Search City like").grid(row=0)
Tkinter.Label(frame, text="User Id").grid(row=1)

e1 = Tkinter.Entry(frame)
e2 = Tkinter.Entry(frame)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

Tkinter.Button(frame, text='Show', command=show_entry_fields).grid(row=3, column=1, sticky=Tkinter.W, pady=4)

canvas.pack(side=Tkinter.LEFT)
Tkinter.mainloop( )

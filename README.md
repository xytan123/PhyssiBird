PhyssiBird
==========
##Demo
https://www.youtube.com/watch?v=Ep8aZGkzRHI

##Intro
A mac game that capture user gesture to enable tangible interaction with the Angry bird game   
I used ```penCV``` to detect user hands frame by frame.

Then we have two points

             X    
           / | --> ^ 
          /  |     | vertical difference -- vertical acceleration   
        X/___| --> |    
        |    |    
        |    |  
        v    v  
        ----->    
        horizontal difference -- horizontal acceleration
           
 Then we can use physics to draw dots along prediction routes.


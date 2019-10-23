def generate_maze(width, height, blockSize=20, basename="maze"):
    '''
    NEVER FORGET YOUR COMMENTS!
    '''
    stack = [(0, 0)]
    maze = [[WALL for _ in range(width)] for _ in range(height)]

    # Set starting position
    x, y = stack[-1]
    maze[y][x] = PATH

    # Set directions - N,S,E,W
    direction_y = [0,0,1,-1] 
    direction_x = [-1,1,0,0]

    # while stack != []:
    while len(stack) > 0:
        # Find all valid moves, and save them in a list.
        x,y=stack[-1]
        n = []
        # First look into "NSEW" directions
        for i in range(4):
            x1 = x+direction_x[i]
            y1 = y+direction_y[i]
#             print("next",y1,x1)
            # check if the position after the move is in the maze
            if x1 >= 0 and x1 < width and y1 >= 0 and y1 < height: 
                # check if it is the wall (can build a path)
                if maze[y1][x1] == WALL:
                    condition=0
                    # check if it will connect to the path
                    # check it will be neibourgh of more than one path
                    for j in range(4):
                        x2 = x1+direction_x[j]
                        y2 = y1+direction_y[j]
                        if x2 >= 0 and x2 < width and y2 >= 0 and y2 < height:
#                             print((y2,x2),maze[y2][x2] == PATH)
                            if maze[y2][x2] == PATH:
                                condition += 1
#                     print("condition",condition)
                    if condition==1:
                        n.append(i)    
        # Randomly take one from the list of valid moves
        if len(n)>0:
            i = random.randint(0,len(n)-1)
        # Append new position to stack
            x += direction_x[n[i]]
            y += direction_y[n[i]]
#             print("true",(y,x))
            stack.append((x,y))
            # change the current position to PATH
            maze[y][x]=PATH
        # If no valid moves exists, backtrack (pop)
        else:
#             stack.pop()
            return maze
#             y,x = stack[-1]
#     return maze
import numpy as np




def LOS(x1, y1, x2, y2):
    """returns a list of all the tiles in the straight line from (x1,y1) to (x2, y2)"""
    point_in_LOS = []
    y=y1
    x=x1
    dx = x2-x1
    dy = y2-y1

    point_in_LOS.append([x1, y1])

    if(dy<0):
        ystep=-1
        dy=-dy
    else:
        ystep=1

    if dx<0:
        xstep=-1
        dx=-dx
    else:
        xstep=1

    ddy = 2*dy
    ddx = 2*dx

    if(ddx >=ddy):
        errorprev = dx
        error = dx
        for i in range(dx):
            x+=xstep
            error +=ddy
            if error>ddx:
                y+=ystep
                error-=ddx
                if (error+errorprev)<ddx:
                    point_in_LOS.append([x, y-ystep])

                elif (error+errorprev) > ddx:
                    point_in_LOS.append([x-xstep, y])

                else:
                    point_in_LOS.append([x, y-ystep])
                    point_in_LOS.append([x-xstep, y])

            point_in_LOS.append([x,y])
            errorprev=error
    else:
        errorprev = dy
        error = dy
        for i in range(dy):
            y += ystep
            error += ddx
            if error>ddy:
                x+=xstep
                error -=ddy
                if (error+errorprev)<ddy:
                    point_in_LOS.append([x-xstep, y])

                elif (error+errorprev)>ddy:
                    point_in_LOS.append([x, y-ystep])

                else:
                    point_in_LOS.append([x, y-ystep])
                    point_in_LOS.append([x-xstep, y])
            point_in_LOS.append([x,y])
            errorprev=error

    return point_in_LOS


def bresenham(x0,y0,x1,y1):
    line = []
    dx = np.abs(x1-x0)
    dy = np.abs(y1-y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx/2.0
        while x!=x1:
            line.append([x,y])
            err-=dy
            if err < 0:
                y+=sy
                err +=dx
            x+=sx
    else:
        err = dy/2.0
        while y!=y1:
            line.append([x,y])
            err -=dx
            if err < 0 :
                x+=sx
                err+=dy
            y+=sy
    line.append([x,y])

    return line


if __name__ == '__main__':
    print(LOS(1, 1, 9, 2))
    print(LOS(9, 2, 1, 1))
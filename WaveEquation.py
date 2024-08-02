import pygame as pg
import numpy as np
def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))
"""Simulation parameters"""
N = 400
L = 1
dx = L/N
pml_width = 40 #perfectly matched layer with for open boundary
dt = 0.001
Tmax = 10
c = -1
obstacle_radius = L/14

is_left_boundary_open = False
is_top_boundary_open = True
is_bottom_boundary_open = True
is_right_boundary_open = True

used_Nx = N + (is_left_boundary_open + is_right_boundary_open)*pml_width
used_Ny = N + (is_top_boundary_open + is_bottom_boundary_open)*pml_width
X, Y = np.mgrid[:used_Nx , :used_Ny].astype(float)*dx
X -= dx*pml_width*is_left_boundary_open
Y -= dx*pml_width*is_top_boundary_open

#obstacle = sigmoid(-1000 * ((X-L/2)**2 + (Y-L/2)**2 - (L/5)**2))
obstacle = 1 - np.exp( -3/dx * (np.maximum(0, (obstacle_radius)**2 - (X-L/2)**2 - (Y-L/2)**2 )))
pml_dist = 1 - np.exp( -1/dx * ((np.minimum(0, X) + np.maximum(L, X) - L)**2 + (np.minimum(0, Y) + np.maximum(L, Y) - L)**2))

"""Displaying parameters"""
display_ratio = 1
preview_averager_ratio = 0.001
def grayify(array, minmax = 1):
    maxabsval = max(np.max(np.abs(array)), 1)
    arr = np.empty((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    arr[:, :, 0] = (array + maxabsval) / (2 * maxabsval) * 255
    arr[:, :, 1] = arr[:, :, 0]
    arr[:, :, 2] = arr[:, :, 0]
    return arr

"""Initialisation"""
U = np.exp(-100*((0.5-X)**2+(0.5-Y)**2))*0
preview = np.zeros_like(U)
U_old = np.copy(U)
def leftValues(x,y,t):
    return (np.sin(t*139))*(np.abs(y-0.5)<0.3)

"""Simulation and display"""
pg.init()
window = pg.display.set_mode((display_ratio*used_Nx, display_ratio*used_Ny))
clock = pg.time.Clock()
t = 0
while t < Tmax:
    """Simulating"""
    Unew = np.copy(U)
    Unew += (1 - 0.1 * pml_dist - 0.1 * obstacle) * (U - U_old)

    Unew[1:-1, :] += (c * dt ** 2 / dx ** 2) * (2 * U[1:-1, :] - (U[0:-2, :] + U[2:, :]))
    Unew[:, 1:-1] += (c * dt ** 2 / dx ** 2) * (2 * U[:, 1:-1] - (U[:, 0:-2] + U[:, 2:]))

    Unew[:, 0] += (c * dt ** 2 / dx ** 2) * (2 * U[:, 0] - (2 * U[:, 1]))
    Unew[:, -1] += (c * dt ** 2 / dx ** 2) * (2 * U[:, -1] - (2 * U[:, -2]))

    Unew[0, :] += (c * dt ** 2 / dx ** 2) * (2 * U[0, :] - (leftValues(X[0], Y[0], t) + U[1, :]))
    #Unew[0, :] += (c * dt ** 2 / dx ** 2) * (2 * U[0, :] - (2 * U[1, :]))
    Unew[-1, :] += (c * dt ** 2 / dx ** 2) * (2 * U[-1, :] - (2*U[-2, :]))
    #Unew[-1, :] += (c * dt ** 2 / dx ** 2) * (2 * U[-1, :] - (2 * U[-2, :]))

    U_old = np.copy(U)
    U = np.copy(Unew)
    preview += preview_averager_ratio*(np.abs(U) - preview)
    """Displaying"""
    t += dt

    for event in pg.event.get():
        if event.type == pg.QUIT:
            t = Tmax

    pg.display.set_caption("t = " + str(np.round(t, 3)))

    #window.blit(pg.transform.smoothscale(pg.surfarray.make_surface(grayify(preview)), (used_Nx*display_ratio, used_Ny*display_ratio)), (0, 0))
    window.blit(pg.transform.smoothscale(pg.surfarray.make_surface(grayify(U)), (used_Nx * display_ratio, used_Ny * display_ratio)), (0, 0))

    pg.draw.rect(window, (0, 100, 255), (display_ratio*pml_width*is_left_boundary_open, display_ratio*pml_width*is_top_boundary_open, display_ratio*N, display_ratio*N), 3)
    pg.draw.circle(window, (0,0,0), (display_ratio*(pml_width*is_left_boundary_open+N/2) , display_ratio*(pml_width*is_top_boundary_open + N/2)), display_ratio*obstacle_radius/dx)
    pg.display.flip()

    clock.tick(120)
pg.quit()





# def action_set(node):
#     r = 5
#     theta_deg = np.linspace(0, 360, 12, endpoint=False)
#     theta     = np.deg2rad(theta_deg)
#     actions = []
#     for i in range(12):
#         x = int(node[0] + r * np.cos(theta[i]))
#         y = int(node[1] + r * np.sin(theta[i]))
#         dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
#         actions.append([(x, y, theta_deg[i]), r])

#     return actions
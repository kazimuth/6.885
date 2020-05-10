using GeometryBasics

"""Inclusive bounds."""
Bounds = Tuple{Float64, Float64}

"""Locations."""
Point2d = Point2{Float64}

"""Velocities / forces / accelerations."""
Vec2d = Vec2{Float64}

"""An array of positions, indexed [timestep, i]"""
PositionArray = Array{Vec2d, 2}

# Create an equally-spaced grid with npts points on the interval [xlow,xhigh].

function creategrid(xlow,xhigh,npts)

   xinc = (xhigh - xlow)/(npts-1)
   
   grid = xlow:xinc:xhigh
   
   gridarray = fill(0.0,npts)
   gridarray .= grid
   
   return gridarray
   
end

# Create an equally-spaced grid with npts points on the interval [xlow,xhigh].

function creategrid(xlow,xhigh,npts)

   xinc = (xhigh - xlow)/(npts-1)

   grid = fill(0.0,npts)
   
   grid[1] = xlow
   
   for i in 2:npts
      grid[i] = grid[i-1] + xinc
   end   

   return grid
   
end

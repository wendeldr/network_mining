data=data_org;
% soz = soz_org.soz(:);
% soz = grp2idx(soz); % 1==false; 2==True
data(sum(isnan(data_org), 2) > 0, :) = [];
% soz(sum(isnan(data_org), 2) > 0, :) = [];


minx = min(data(:,1));
maxx = max(data(:,1));
ux = unique(data(:,1));

% y = ones(size(ux));
% scatter(ux,y,1,'k','filled')
% xline(minx)
% xline(maxx)

miny = min(data(:,2));
maxy = max(data(:,2));
uy = unique(data(:,2));
% y = ones(size(uy));
% scatter(uy,y,1,'k','filled')
% xline(miny)
% xline(maxy)

minz = min(data(:,3));
maxz = max(data(:,3));
uz = unique(data(:,3));
% y = ones(size(uz));
% scatter(uz,y,1,'k','filled')
% xline(minz)
% xline(maxz)

% scatter3(ux,uy,uz,1,'k','filled')

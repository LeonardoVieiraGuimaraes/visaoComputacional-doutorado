clear all;

rgb = imread('I_0h_1.jpg');

%hsv = rgb2hsv(rgb);

imshow(rgb);
d = imdistline;
delete(d);

gray = rgb2gray(rgb);

imshow(gray);

[centers, radii] = imfindcircles(rgb,[20 60], 'objectPolarity','dark', 'Sensitivity',0.9);

imshow(rgb);

h = viscircles(centers, radii);



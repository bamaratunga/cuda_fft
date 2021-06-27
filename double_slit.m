N2 = 16;
[gx, gy] = meshgrid( -1: 1/N2: (N2-1)/N2 );

% aperture       = ( abs(gx) < 4/N2 ) .* ( abs(gy) < 2/N2 );
% lightsource    = double( aperture );
% farfieldsignal = fft2( lightsource );
% 
% farfieldintensity = real( farfieldsignal .* conj( farfieldsignal ) );
% 
% imagesc( fftshift( farfieldintensity ) );
% axis( 'equal' ); axis( 'off' );
% title( 'Rectangular aperture far-field diffraction pattern' );

width = 2;
height = 5;
dist = 8;

% for dist = 1:20
    slits          = (abs( gx ) <= (dist+width)/N2) .* (abs( gx ) >= dist/N2);
    aperture       = slits .* (abs(gy) < height/N2);
    lightsource    = double( aperture );
    farfieldsignal = fft2( lightsource );


    farfieldintensity = real( farfieldsignal .* conj( farfieldsignal ) );
    imagesc( fftshift( farfieldintensity ) );
    axis( 'equal' ); axis( 'off' );
    title( 'Double slit far-field diffraction pattern' );
    pause(1.0);
% end
/****************************************************************************
 *
 * cuda-cat-map.c - Arnold's cat map with CUDA
 *
 * Written in 2016, 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 *
 * nvcc cuda-cat-map.cu -o cuda-cat-map
 *
 * Run with:
 *
 * ./cuda-cat-map k < input_file > output_file
 *
 * to compute the k-th iterate of the cat map.  Input and output files
 * are in PGM (Portable Graymap) format; see "man pgm" for details
 * (however, you do not need to know anything about the PGM formato;
 * functions are provided below to read and write a PGM file). The
 * file cat.pgm can be used as a test image.  Example:
 *
 * ./cuda-cat-map 100 < cat.pgm > cat.100.pgm
 *
 * See https://en.wikipedia.org/wiki/Arnold%27s_cat_map for an explanation
 * of the cat map.
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width*height bytes; each byte represents a pixel */
} img_t;

/**
 * Read a PGM file from standard input. This function is not very
 * robust; it may fail on perfectly legal PGM images, but works for
 * the provided cat.pgm file. 
 */
void read_pgm( img_t* img )
{
    /* Since the PGM file includes binary data right after the ASCII
       header, we must read the header one line at a time (including
       the trailing newline), so that the newline character does not
       appear within the binary data */
    char buf[255]; 
    char *s; 
    int nread;

    /* Get the file type (must be "P5") */
    s = fgets(buf, sizeof(buf), stdin);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(-1);
    }
    /* Get the comment and ignore it */
    s = fgets(buf, sizeof(buf), stdin);
    /* Get width, height */
    s = fgets(buf, sizeof(buf), stdin);
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, sizeof(buf), stdin);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "Error: maxgray > 255 (%d)\n", img->maxgrey);
        exit(-1);
    }
    /* Get the binary data */
    img->bmap = (unsigned char*)malloc((img->width)*(img->height));
    nread = fread(img->bmap, 1, (img->width)*(img->height), stdin);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "Error reading input file: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(-1);
    }
}

/**
 * Write image |img| to standard output 
 */
void write_pgm( const img_t* img )
{
    printf("P5\n");
    printf("# produced by cuda-cat-map\n");
    printf("%d %d\n", img->width, img->height);
    printf("%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), stdout);
}

/**
 * Compute the |k|-th iterate of the cat map for image |img|. You must
 * implement this function, starting with a serial version, and then
 * adding OpenMP pragmas. This function must replace the bitmap of
 * |img| with the one resulting after ierating |k| times the cat
 * map. You need to allocate a temporary image, with the same size of
 * the original one, so that you read the pixel from the "old" image
 * and copy them to the "new" image (this is similar to a stencil
 * computation, as was discussed in class). After applying the cat map
 * to all pixel of the "old" image the role of the two images is
 * exchanged: the "new" image becomes the "old" one, and
 * vice-versa. At the end of the function, the temporary image must be
 * deallocated. 
 */
void cat_map( img_t* img, int k )
{
    int i, x, y;
    const int w = img->width;
    const int h = img->height;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char*)malloc( w*h*sizeof(*next) );
    unsigned char *tmp;

    for (i=0; i<k; i++) {
        for (y=0; y<h; y++) {
            for (x=0; x<w; x++) {
                int xnext = (2*x+y) % w;
                int ynext = (x + y) % h;
                next[xnext + ynext*w] = cur[x+y*w];
            }
        }
        /* Swap old and new */
        tmp = cur;
        cur = next;
        next = tmp;
    }
    img->bmap = cur;
    free(next);
}

int main( int argc, char* argv[] )
{
    img_t bmap;
    int niter;
    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < input_file > output_file\n", argv[0]);
        return -1;
    }
    niter = atoi(argv[1]);
    read_pgm(&bmap);
    double tstart = hpc_gettime();
    cat_map(&bmap, niter);
    double tend = hpc_gettime();
    fprintf(stderr, "Execution time %f\n", tend - tstart);
    write_pgm(&bmap);
    return 0;
}

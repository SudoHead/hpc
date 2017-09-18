/****************************************************************************
 *
 * omp-brute-force.c - Brute-force password cracking
 *
 * Written in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * --------------------------------------------------------------------------
 *
 * Skeleton program containing an encrypted message that must be
 * decrypted by brute-forcing the key space using OpenMP
 * costructs. The encryption key is a sequence of 8 ASCII numeric
 * charactgers, therefore belongs to the interval "00000000" -
 * "99999999"; the correctly decrypted message is a sequence of
 * printable characters and starts with "0123456789" (no quotes).
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <rpc/des_crypt.h>

void decrypt(const char* enc, char* dec, int n, const char* key)
{
    int err;
    char keytmp[8];
    assert( n % 8 == 0 );   /* ecb_crypt requires the data length to be a multiple of 8 */
    memcpy(keytmp, key, 8); /* copy the key to a temporary buffer, since  des_setparity() will modify it */
    memcpy(dec, enc, n);    /* copy the encrypted message to the decription buffer */
    des_setparity(keytmp);  /* set key parity */
    err = ecb_crypt(keytmp, dec, n, DES_DECRYPT | DES_SW);
    assert( DESERR_NONE == err );
}

int main( int argc, char *argv[] )
{
    const char enc[] = {-87,12,111,22,108,-1,122,73,6,53,64,33,12,52,70,-53,-37,74,45,-4,121,-102,34,-56,23,-32,113,2,4,-5,-119,65,-112,94,-34,52,-34,-42,60,-5,-88,-28,118,-97,-11,-40,20,97,125,-93,-58,-1,-30,120,52,-93,-107,-101,-21,-41,-20,10,99,-98};
    const int msglen=64; /* Length of the encrypted message */

    /* How to use a key to decrypt the message */
    char key[9]; 
    int k = 132; /* numeric value of the key to try */
    char* out = (char*)malloc(msglen);
    snprintf(key, 9, "%08d", k);
    decrypt(enc, out, msglen, key);

    return 0;
}

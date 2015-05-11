from PIL import Image
import itertools
import numpy as np
import json

def generate_all_markers(inner, outer, size, max_range=100):
    ''' Returns a PNG Image Marker generated using PIL'''

    im = Image.new('1', size)
    px = im.load()

    dim = outer + inner + outer
    nb_combi = 2**(inner*inner)
    mots = np.array(list(itertools.product((0, 1), repeat=inner**2)))
    all_bit_matrix = [mot.reshape((inner, inner)) for mot in mots]
    unique_matrices = []
    for i in range(len(all_bit_matrix)):
        rmat = all_bit_matrix[i]
        for j in range(3):
            rmat = np.rot90(rmat)
            if np.array_equal(all_bit_matrix[i], rmat):
                break
        else:
            marker_mat = np.array([[0 for l in range(dim)] for c in range(dim)])
            marker_mat[outer:outer+inner,outer:outer+inner] = all_bit_matrix[i]
            unique_matrices.append(marker_mat)

    unique_markers = []
    for i in range(len(unique_matrices)):
        rmat = unique_matrices[i]
        for j in range(3):
            rmat = np.rot90(rmat)
            if any(np.array_equal(mat, rmat) for mat in unique_markers):
                break
        else:
            unique_markers.append(unique_matrices[i])

    #print('nb_combi={}, unique_matrices={}, unique_markers={}'.format(nb_combi, len(unique_matrices), len(unique_markers)))
    unique_markers = unique_markers[:min(max_range, len(unique_markers))]
    for i in range(len(unique_markers)):
        w, h = size
        col_index, line_index = -1, -1
        color = unique_markers[i][line_index][col_index]
        for x in range(w):
            line_index = 0
            if x % (w//dim) == 0:
                col_index += 1
            for y in range(h):
                if y % (h//dim) == 0:
                    color = unique_markers[i][line_index][col_index]
                    line_index += 1
                px[x, y] = color
        im.save('../media/markers/marker_{}.png'.format(i+1), 'PNG')
    return unique_markers


def main():
    inner, outer = 2, 2 # 7x7 square matrix (5x5 codable)
    img_size = 300, 300 # 280x280 px image
    max_markers = 100
    print('>> Generate {0}x{0} markers (inner={1}, outer={2}, pxsize={3}px)'.format(
    outer+inner+outer, inner, outer, img_size))
    bit_matrices = generate_all_markers(inner, outer, img_size, max_markers)
    print('>> {} markers has been created'.format(len(bit_matrices)))
    print('>> Create JSON with {} created markers'.format(len(bit_matrices)))
    fic = open("ref_markers.json", 'w')
    bit_matrices = [m.tolist() for m in bit_matrices]
    fic.write(json.dumps(list(bit_matrices)))
    fic.close()
    print('>> JSON done (markers.json)')

if __name__ == '__main__':
    main()

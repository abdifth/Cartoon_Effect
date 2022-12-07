from skimage import io
import argparse
import cartoon_effect

# def apply(img, sig_b, sig_x, p, N, T):

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--dog", type=str, default = "dog.jpg")
    parser.add_argument("--xdog", type=str, default = "xdog.jpg")
    parser.add_argument("--simage", type=str, default = "simage.jpg")
    parser.add_argument("--sig_b", type=str, default = 1.1) #1.1
    parser.add_argument("--sig_x", type=str, default = 0.8)
    parser.add_argument("--p", type=str, default = 4)
    parser.add_argument("--N", type=str, default = 4)
    parser.add_argument("--T", type=str, default = 0.28)
    args = parser.parse_args()
    input_image = io.imread(args.input)
    result,xdog,dog,simage = cartoon_effect.CartoonEffect.apply(input_image,args.sig_b,args.sig_x,args.p,args.N,args.T)

    io.imsave(args.output,result)
    io.imsave(args.xdog,xdog)
    io.imsave(args.dog,dog)
    io.imsave(args.simage,simage)


if __name__ == '__main__':
    main()

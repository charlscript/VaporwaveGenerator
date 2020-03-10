import sys
import os
import logging

import cv2

from vaporwave import vaporize

ESCAPE_KEY = 27

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def main():
    path = ""

    if len(sys.argv)>1 and os.path.exists(sys.argv[1]):
        path = os.path.abspath(sys.argv[1])
    else:
        print("Passe uma imagem png como argumento")
        exit()
    img = vaporize(path)


    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    cv2.imshow("pic", img)

    while cv2.getWindowProperty("pic", cv2.WND_PROP_VISIBLE):
        key_code = cv2.waitKey(100)

        if key_code == ESCAPE_KEY:
            break
        elif key_code != -1:
            import time
            start = time.time()
            img = vaporize(path)
            cv2.imshow("pic", img)
            end = time.time()
            logger.info("O processo de vaporwarização levou: %f seconds" % (end-start,))
    cv2.destroyAllWindows()
    sys.exit()


if __name__ == "__main__":
    main()

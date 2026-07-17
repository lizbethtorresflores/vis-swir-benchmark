import os
import cv2
import argparse
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import message_to_cvimage

TARGET_TOPICS = ["/camera_SWIR", "/camera_AV"]


def save_image_with_metadata(base_dir, topic_name, seq, timestamp, img, output_format="tiff"):
    clean_topic_name = topic_name.strip("/").replace("/", "_")
    topic_dir = os.path.join(base_dir, clean_topic_name)
    os.makedirs(topic_dir, exist_ok=True)

    seq_padded    = str(seq).zfill(6)
    secs          = timestamp // 10**9
    nsecs         = timestamp % 10**9
    timestamp_str = f"{secs}_{nsecs:09d}"

    filename = f"seq_{seq_padded}_ts_{timestamp_str}.{output_format}"
    img_path = os.path.join(topic_dir, filename)
    cv2.imwrite(img_path, img)
    return img_path

def find_metadata_yaml(root: Path) -> Path | None:
    direct = root / "metadata.yaml"
    if direct.exists():
        return root

    candidates = []
    for sub in root.iterdir():
        if sub.is_dir():
            meta = sub / "metadata.yaml"
            if meta.exists():
                candidates.append(sub)

    if len(candidates) == 1:
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Extrait les images d'un bag ROS2")
    parser.add_argument("bag", help="Chemin relatif à datas/ (dossier ou .zip)")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--format", "-f", default="tiff")
    args = parser.parse_args()

    projet_root = Path(__file__).resolve().parents[3]
    bag_path    = (projet_root / "datas" / args.bag).resolve()

    if not bag_path.exists():
        print(f"[ERREUR] Bag introuvable : {bag_path}")
        raise SystemExit(1)

    if bag_path.suffix == ".zip":
        import zipfile
        extract_dir = bag_path.parent / bag_path.stem
        if not extract_dir.exists():
            print(f"Décompression de {bag_path.name} → {extract_dir}")
            with zipfile.ZipFile(bag_path, 'r') as zf:
                zf.extractall(extract_dir)
        else:
            print(f"Dossier déjà extrait : {extract_dir}")
        bag_path = extract_dir

    bag_root = find_metadata_yaml(bag_path)
    if bag_root is None:
        print(f"[ERREUR] metadata.yaml introuvable sous {bag_path}")
        raise SystemExit(1)
    if bag_root != bag_path:
        bag_path = bag_root
        print(f"Bag ajusté : {bag_path}")

    output_path = Path(args.output).resolve() if args.output else bag_path.parent / "Sortie_Extraite"

    print(f"Bag    : {bag_path}")
    print(f"Sortie : {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.LATEST)
    count = 0

    with Reader(str(bag_path)) as reader:
        print("\nTopics disponibles :")
        for connection in reader.connections:
            print(f"  - {connection.topic} [{connection.msgtype}]")
        print()

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic in TARGET_TOPICS:
                try:
                    msg  = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    img  = message_to_cvimage(msg)
                    path = save_image_with_metadata(
                        str(output_path), connection.topic,
                        count, timestamp, img, output_format=args.format
                    )
                    count += 1
                    if count % 10 == 0:
                        print(f"{count} images extraites...", end='\r')
                    if count == 1:
                        print(f"\n[DEBUG] Première image : {path}")
                        print(f"        Shape: {img.shape}, dtype: {img.dtype}")

                except Exception as e:
                    print(f"\n[ERREUR] {connection.topic} : {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\nTerminé ! {count} images extraites")

if __name__ == '__main__':
    main()

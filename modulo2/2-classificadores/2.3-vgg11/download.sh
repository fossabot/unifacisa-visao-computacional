# Peso pré-treianados
mkdir pesos && cd pesos
wget https://download.pytorch.org/models/vgg11-bbd30ac9.pth
cd ..

# Labels
mkdir labels && cd labels
wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
cd ..

# Baixa imagem para teste
mkdir imagens && cd imagens
wget https://myanimals.com/pt/wp-content/uploads/2015/08/raposa.jpg
wget https://i.pinimg.com/originals/61/c0/d1/61c0d126d3654afee7c0cba0e1cba11f.jpg
cd ..
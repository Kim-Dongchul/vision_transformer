import torch
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid

# def train(train_loader, model, num_epochs, optimizer, criterion, device, logging_step=1, is_img=False):
    # writer = SummaryWriter()
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item() * images.size(0)
    #
    #     _, predicted = torch.max(outputs, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    #     if epoch % logging_step == 0:
    #         print(f'Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / len(train_loader.dataset)}')
        # writer.add_scalar('Loss/train', running_loss / len(train_loader.dataset), epoch)
    # if is_img:
    #     writer.add_image('Images/train', make_grid(images))
    # writer.add_graph(model, images)
    #
    # writer.close()

def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 정확도 계산
        total += labels.size(0)
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy

# 평가 함수 정의
def evaluate(model, test_loader, criterion, device):
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 중에는 기울기 계산을 하지 않음
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 예측 정확도 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy

# 학습 및 평가 과정 관리
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, logging_step=1):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        # 모델 학습(학습데이터)
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if test_loader is not None:
            # 모델 평가 (평가데이터)
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        if epoch % logging_step == 0:
            if test_loader is not None:
                print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.2f}% Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.2f}%')
            else:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.2f}%')

    return train_losses, train_accuracies, test_losses, test_accuracies

# def evaluate(test_loader, model, device, s_idx):
#     model.eval()
#     results = []
#     file_path = f'summit/{s_idx}.csv'
#     with torch.no_grad():
#         for idx, (images, _) in enumerate(test_loader):
#             images = images.to(device)
#             outputs = model(images).logits
#             probs = torch.softmax(outputs, dim=1)
#             results.append([f'Test_{idx}'] + probs.squeeze().cpu().numpy().tolist())
#
#     results_df = pd.DataFrame(results, columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
#     results_df.to_csv(file_path, index=False)


def train_logit(model, train_loader, criterion, optimizer, device):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 정확도 계산
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy

# 평가 함수 정의
def evaluate_logit(model, test_loader, criterion, device):
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 중에는 기울기 계산을 하지 않음
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 예측 정확도 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy

# 학습 및 평가 과정 관리
def train_and_evaluate_logit(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, logging_step=1):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        # 모델 학습(학습데이터)
        train_loss, train_accuracy = train_logit(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 모델 평가 (평가데이터)
        test_loss, test_accuracy = evaluate_logit(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if epoch % logging_step == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.2f}% Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.2f}%')

    return train_losses, train_accuracies, test_losses, test_accuracies
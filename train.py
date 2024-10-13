folder_path = "192-48-48"

train_dl_list = [train_dl, train_dl1, train_dl2]
test_dl_list = [test_dl, test_dl1, test_dl2]

for epoch in range(num_epochs):
    for dl_idx, (train_dl, test_dl) in enumerate(zip(train_dl_list, test_dl_list)):
        y_train_pred_list = []
        y_train_true_list = []
        sum_loss = 0
        
        model.train()
        
        # 训练集
        for bx, by in train_dl:
            if use_gpu:
                bx, by = bx.cuda(), by.cuda()

            y_train = model(bx)
            loss = criterion(y_train, by)
            # 归一化
            # y_train = y_train * y_range + y_min 
            # by = by * y_range + y_min 

            y_train_pred_list.append(y_train)
            y_train_true_list.append(by)

            if use_gpu:
                loss = loss.cpu()
            sum_loss += loss
            
            loss = criterion(y_train, by)
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            loss.backward()
            optimizer.step()
        scheduler.step() 
        
        avg_loss_epoch = sum_loss / len(train_dl)
        print(f'Model {dl_idx + 1}, Epoch: {epoch + 1}, Train Loss: {avg_loss_epoch:.5f}')

    if((epoch+1) % 5 == 0):       

        y_train_pred = torch.cat(y_train_pred_list, dim=0)
        y_train_pred = y_train_pred.view(train_num, step).cpu().detach().numpy()     
        
        y_train_true = torch.cat(y_train_true_list, dim=0)
        y_train_true = y_train_true.view(train_num, step).cpu().detach().numpy() 

        model.eval()
        
        # 正常的
        y_test_pred_list = []
        y_test_true_list = []
        
        # 反归一化后的
        y_test_pred_list_2 = []
        y_test_true_list_2 = []
        
        with torch.no_grad():
            sum_test_loss = 0
            for bx, by in test_dl:  
                bx,by = bx.cuda(),by.cuda()
                
                y_test_pred = model(bx)

                y_test_true_list.append(by)
                y_test_pred_list.append(y_test_pred)
                
                # 反归一化
                y_test_pred_2 = y_test_pred * y_range + y_min 
                by_2 = by *  y_range + y_min 
                
                # 反归一化
                y_test_true_list_2.append(by_2)
                y_test_pred_list_2.append(y_test_pred_2)
                
                if use_gpu:
                    loss = loss.cpu()
                sum_test_loss += loss
            avg_loss_epoch = sum_test_loss / len(test_dl)
            
            print(f'Epoch: {epoch + 1}, Test Loss: {avg_loss_epoch:.5f}')
         
            y_test_pred = torch.cat(y_test_pred_list, dim=0)
            y_test_pred = y_test_pred.view(test_num, step).cpu().detach().numpy() 
        
            y_test_true = torch.cat(y_test_true_list, dim=0)
            y_test_true = y_test_true.view(test_num, step).cpu().detach().numpy() 
            
            # 反归一化
            y_test_pred_2 = torch.cat(y_test_pred_list_2, dim=0)
            y_test_pred_2 = y_test_pred_2.view(test_num, step).cpu().detach().numpy() 
        
            y_test_true_2 = torch.cat(y_test_true_list_2, dim=0)
            y_test_true_2 = y_test_true_2.view(test_num, step).cpu().detach().numpy() 
            
            # pd.DataFrame(y_test_pred_2).to_csv(f"sp500(1924848)-test{epoch + 1}.csv",index=False)
            # pd.DataFrame(y_test_true_2).to_csv(f"sp500(1924848)-test-true{epoch + 1}.csv",index=False)
            # 保存预测结果到CSV文件
            pd.DataFrame(y_test_pred_2).to_csv(os.path.join(folder_path, f"btc(1924848)-test{epoch + 1}.csv"), index=False)
            # 保存真实结果到CSV文件
            pd.DataFrame(y_test_true_2).to_csv(os.path.join(folder_path, f"btc(1924848)-test-true{epoch + 1}.csv"), index=False)
                                               
            for m in range(1,33):
                y_test_pred_m = y_test_pred[:, m-1]
                y_test_true_m = y_test_true[:, m-1]
                print(f"R2 {m}:",r2_score(y_test_true_m, y_test_pred_m))
                print(f"MAE {m}:", mean_absolute_error(y_test_true_m, y_test_pred_m))
                print(f"RMSE {m}:", mean_squared_error(y_test_true_m, y_test_pred_m))
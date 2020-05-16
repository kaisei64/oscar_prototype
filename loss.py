lambda_class = 20
lambda_ae = 1
lambda_1 = 1
lambda_2 = 1


def prototype_loss(class_error, ae_error, error_1, error_2, error_1_flag=True, error_2_flag=True):
    if error_1_flag is True and error_2_flag is True:
        return lambda_class * class_error + lambda_ae * ae_error + lambda_1 * error_1 + lambda_2 * error_2
    elif error_1_flag is True and error_2_flag is False:
        return lambda_class * class_error + lambda_ae * ae_error + lambda_1 * error_1
    elif error_1_flag is False and error_2_flag is True:
        return lambda_class * class_error + lambda_ae * ae_error + lambda_2 * error_2
    else:
        return lambda_class * class_error + lambda_ae * ae_error

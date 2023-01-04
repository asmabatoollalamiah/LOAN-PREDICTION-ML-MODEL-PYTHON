import os;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from scipy import stats;
import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import classification_report;
from sklearn.metrics import confusion_matrix;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import MinMaxScaler;
from imblearn.over_sampling import SMOTE;
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

class LoanPredictionOperations():
    
    def readCSVDatatoDataFrame(self):
        '''This method Reads the CSV file and store into Data Frame'''
        df_csv_loan_data_of_customer = pd.DataFrame();
        try:
          if(os.path.exists('/content/drive/My Drive/Python_Final_Files/Customer_Details_For_Loan.csv')):
            df_csv_loan_data_of_customer = pd.read_csv('/content/drive/My Drive/Python_Final_Files/Customer_Details_For_Loan.csv');
          else:
            print("Customer_Details_For_Loan.csv file not found");
        except FileNotFoundError:
            print('File not found Error!!:: Customer_Details_For_Loan.csv');
        except Exception:
            print('Error occured reading  Customer_Details_For_Loan.csv File!!');
        return df_csv_loan_data_of_customer;
        
        
    def describeNumericalVairables(self, df_csv_loan_data_of_customer):
        '''This function displays the describe of Numberical Variables'''
        try:
            print(df_csv_loan_data_of_customer[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']].describe());
        except Exception:
            print('Error occured in describeNumericalVairables');
        
    def displayEmptyColCount(self, df_csv_loan_data_of_customer):
        '''This function displayes the empty column count'''
        try:
            print(df_csv_loan_data_of_customer.isna().sum().where(lambda x:x>0).dropna());
        except Exception:
            print('Error occured in displayEmptyColCount');
        
    def fillCategoricalColumnsWithMode(self, df_csv_loan_data_of_customer):
        '''This function fills the Categorical Columns with mode value the Data Frame'''
        try:
            df_csv_loan_data_of_customer['Gender'].fillna(df_csv_loan_data_of_customer['Gender'].mode()[0],inplace=True)
            df_csv_loan_data_of_customer['Married'].fillna(df_csv_loan_data_of_customer['Married'].mode()[0],inplace=True)
            df_csv_loan_data_of_customer['Dependents'].fillna(df_csv_loan_data_of_customer['Dependents'].mode()[0],inplace=True)
            df_csv_loan_data_of_customer['Self_Employed'].fillna(df_csv_loan_data_of_customer['Self_Employed'].mode()[0],inplace=True)
            df_csv_loan_data_of_customer['Credit_History'].fillna(df_csv_loan_data_of_customer['Credit_History'].mode()[0],inplace=True)
            df_csv_loan_data_of_customer['Loan_Amount_Term'].fillna(df_csv_loan_data_of_customer['Loan_Amount_Term'].mode()[0],inplace=True)
        except Exception:
            print('Error occured in fillCategoricalColumnsWithMode');
        return df_csv_loan_data_of_customer;
    
    
    def fillNumericalColumnsWithMeanValue(self, df_csv_loan_data_of_customer):
        '''This function fills the Categorical Columns with mode value the Data Frame'''
        try:
            df_csv_loan_data_of_customer['LoanAmount'].fillna(df_csv_loan_data_of_customer['LoanAmount'].mean(),inplace=True)
        except Exception:
            print('Error occured in fillNumericalColumnsWithMeanValue');
        return df_csv_loan_data_of_customer;
    
    
    def getGenderValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Gender Value Count'''
        gender_val_count_ser = pd.Series();
        try:
            gender_val_count_ser = df_csv_loan_data_of_customer.Gender.value_counts();
        except Exception:
            print('Error occured in getGenderValueCount');
        return gender_val_count_ser;
         
    def displayApplicantsPercentageByGender(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Gender'''
        try:
            countOfMaleApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Gender == 'Male']);
            countOfFemaleApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Gender == 'Female']);
            print("The Percentage of Male applicants: {:.2f}%".format((countOfMaleApplicants / (len(df_csv_loan_data_of_customer.Gender))*100)));
            print("The Percentage of Female applicants: {:.2f}%".format((countOfFemaleApplicants / (len(df_csv_loan_data_of_customer.Gender))*100)));
        except Exception:
            print('Error occured in displayApplicantsPercentageByGender');
             
         
    def getMarriedValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Married Value Count'''
        try:
            return df_csv_loan_data_of_customer.Married.value_counts();
        except Exception:
            print('Error occured in getMarriedValueCount');
         
         
    def displayApplicantsPercentageByMarried(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Married'''
        try:
            countOfMarriedApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Married == 'Yes']);
            countOfNotMarriedApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Married == 'No']);
            print("The Percentage of Married applicants: {:.2f}%".format((countOfMarriedApplicants / (len(df_csv_loan_data_of_customer.Married))*100)));
            print("The Percentage of not Married applicants: {:.2f}%".format((countOfNotMarriedApplicants / (len(df_csv_loan_data_of_customer.Married))*100)));
        except Exception:
            print('Error occured in displayApplicantsPercentageByMarried');
    
    def getEducationValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Education Value Count'''
        education_val_count_ser = pd.Series();
        try:
            education_val_count_ser = df_csv_loan_data_of_customer.Education.value_counts();
        except Exception:
            print('Error occured in getEducationValueCount');
        return education_val_count_ser; 
         
    def getSelfEmployedValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Self_Employed Value Count'''
        self_employed_val_count_ser = pd.Series();
        try:
            self_employed_val_count_ser = df_csv_loan_data_of_customer.Self_Employed.value_counts();
        except Exception:
            print('Error occured in getSelfEmployedValueCount');
        return self_employed_val_count_ser; 
    
    def displayApplicantsPercentageBySelfEmployed(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Self_Employed'''
        try:
            countOfSelfEmployedApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Self_Employed == 'Yes']);
            countOfNotSelfEmployedApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Self_Employed == 'No']);
            print("The Percentage of Self Employed applicants: {:.2f}%".format((countOfSelfEmployedApplicants / (len(df_csv_loan_data_of_customer.Self_Employed))*100)));
            print("The Percentage of not Self Employed applicants: {:.2f}%".format((countOfNotSelfEmployedApplicants / (len(df_csv_loan_data_of_customer.Self_Employed))*100)));
        except Exception:
            print('Error occured in displayApplicantsPercentageBySelfEmployed');

    def getCreditHistoryValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Credit_History Value Count'''
        credit_history_val_count_ser = pd.Series();
        try:
            credit_history_val_count_ser = df_csv_loan_data_of_customer.Credit_History.value_counts();
        except Exception:
            print('Error occured in getCreditHistoryValueCount');
        return credit_history_val_count_ser;
        
        
    def displayApplicantsPercentageByCreditHistory(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Credit_History'''
        try:
            cntOfGoodCreditHistoryApp= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Credit_History == 1]);
            cntOfBadCreditHistoryApp = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Credit_History == 0]);
            print("The Percentage of Good Credit History applicants: {:.2f}%".format((cntOfGoodCreditHistoryApp / (len(df_csv_loan_data_of_customer.Credit_History))*100)));
            print("The Percentage of Bad Credit History applicants: {:.2f}%".format((cntOfBadCreditHistoryApp / (len(df_csv_loan_data_of_customer.Credit_History))*100)));
        except Exception:
            print('Error occured in displayApplicantsPercentageByCreditHistory'); 
         
    def getPropertyAreaValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Property_Area Value Count'''
        propery_area_val_count_ser = pd.Series();
        try:
            propery_area_val_count_ser = df_csv_loan_data_of_customer.Property_Area.value_counts();
        except Exception:
            print('Error occured in getPropertyAreaValueCount');
        return propery_area_val_count_ser; 
         
    def displayApplicantsPercentageByPropertyArea(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Property_Area'''
        try:
            countOfUrbanApplicants= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Property_Area == 'Urban']);
            countOfRuralApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Property_Area == 'Rural']);
            countOfSemiurbanApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Property_Area == 'Semiurban']);
            print("The Percentage of Urban Property Area applicants: {:.2f}%".format((countOfUrbanApplicants / (len(df_csv_loan_data_of_customer.Property_Area))*100)));
            print("The Percentage of Rural Property Area applicants: {:.2f}%".format((countOfRuralApplicants / (len(df_csv_loan_data_of_customer.Property_Area))*100)));
            print("The Percentage of Semiurban Property Area applicants: {:.2f}%".format((countOfSemiurbanApplicants / (len(df_csv_loan_data_of_customer.Property_Area))*100)));
        except Exception:
            print('Error occured in displayApplicantsPercentageByPropertyArea');
    
    def getLoanStatusValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Loan_Status Value Count'''
        loan_status_val_count_ser = pd.Series();
        try:
            loan_status_val_count_ser = df_csv_loan_data_of_customer.Loan_Status.value_counts();
        except Exception:
            print('Error occured in getLoanStatusValueCount');
        return loan_status_val_count_ser; 
        
    def displayApplicantsPercentageByLoanStatus(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Loan_Status'''
        try:
            countOfApprovedApplicants= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Status == 'Y']);
            countOfRejectedApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Status == 'N']);
            print("The Percentage of approved applicants: {:.2f}%".format((countOfApprovedApplicants / (len(df_csv_loan_data_of_customer.Loan_Status))*100)));
            print("The Percentage of rejected applicants: {:.2f}%".format((countOfRejectedApplicants / (len(df_csv_loan_data_of_customer.Loan_Status))*100)));
        except Exception:
            print('Error occured in displayApplicantsPercentageByLoanStatus');


    def getLoanAmountTermValueCount(self, df_csv_loan_data_of_customer):
        '''This function returns the Loan_Amount_Term Value Count'''
        loan_amount_val_count_ser = pd.Series();
        try:
            loan_amount_val_count_ser =  df_csv_loan_data_of_customer.Loan_Amount_Term.value_counts();
        except Exception:
            print('Error occured in getLoanAmountTermValueCount');
        return loan_amount_val_count_ser; 
    
    def displayAppPercentageByLoanAmountTerm(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Loan_Amount'''
        try:
            countOfLoanAmountTerm12= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 12.0]);
            countOfLoanAmountTerm36= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 36.0]);
            countOfLoanAmountTerm60= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 60.0]);
            countOfLoanAmountTerm84= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 84.0]);
            countOfLoanAmountTerm120= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 120.0]);
            countOfLoanAmountTerm180= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 180.0]);
            countOfLoanAmountTerm240= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 240.0]);
            countOfLoanAmountTerm300= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 300.0]);
            countOfLoanAmountTerm360= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 360.0]);
            countOfLoanAmountTerm480= len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Loan_Amount_Term == 480.0]);
            print("The Percentage of applicants with Loan Amount of term 12 : {:.2f}%".format((countOfLoanAmountTerm12 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 36 : {:.2f}%".format((countOfLoanAmountTerm36 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 60 : {:.2f}%".format((countOfLoanAmountTerm60 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 84 : {:.2f}%".format((countOfLoanAmountTerm84 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 120 : {:.2f}%".format((countOfLoanAmountTerm120 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 180 : {:.2f}%".format((countOfLoanAmountTerm180 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 240 : {:.2f}%".format((countOfLoanAmountTerm240 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 300 : {:.2f}%".format((countOfLoanAmountTerm300 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 360 : {:.2f}%".format((countOfLoanAmountTerm360 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
            print("The Percentage of applicants with Loan Amount of term 480 : {:.2f}%".format((countOfLoanAmountTerm480 / (len(df_csv_loan_data_of_customer.Loan_Amount_Term))*100)));
        except Exception:
            print('Error occured in displayAppPercentageByLoanAmountTerm');
        
        
    def createOneHotEncodingByGetDummies(self, df_csv_loan_data_of_customer):
        '''This function returns a One Hot Encoding of Cateforial values By Get Dummies function'''
        df_csv_loan_data_of_customer_dummies = pd.DataFrame();
        try:
            df_csv_loan_data_of_customer_dummies = pd.get_dummies(df_csv_loan_data_of_customer);
        except Exception:
            print('Error occured in createOneHotEncodingByGetDummies');
        return df_csv_loan_data_of_customer_dummies;
        
    def dropAndRenameColumns(self, df_csv_loan_data_of_customer):
        '''This function drops and renames the columns'''
        try:
            df_csv_loan_data_of_customer = df_csv_loan_data_of_customer.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 'Self_Employed_No', 'Loan_Status_N'], axis = 1);
            newColumnNames = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed', 'Loan_Status_Y': 'Loan_Status'};
            df_csv_loan_data_of_customer.rename(columns=newColumnNames, inplace=True);
        except Exception:
            print('Error occured in dropAndRenameColumns');
        return df_csv_loan_data_of_customer;
        
        
        
    def removeOutliersAndInfiniteValues(self, df_csv_loan_data_of_customer): 
        '''This function removes the Outliers and Infinite Values'''
        try:
            Q1 = df_csv_loan_data_of_customer.quantile(0.25)
            Q3 = df_csv_loan_data_of_customer.quantile(0.75)
            IQR = Q3 - Q1
            df_csv_loan_data_of_customer = df_csv_loan_data_of_customer[~((df_csv_loan_data_of_customer < (Q1 - 1.5 * IQR)) |(df_csv_loan_data_of_customer > (Q3 + 1.5 * IQR))).any(axis=1)]
        except Exception:
            print('Error occured in removeOutliersAndInfiniteValues');
        return df_csv_loan_data_of_customer;
        
        
    def performSquareRootTransformation(self, df_csv_loan_data_of_customer):
        '''This function performs the square root transformation of ApplicantIncome, CoapplicantIncome ,  LoanAmount and Loan_Amount_Term'''
        try:
            df_csv_loan_data_of_customer.ApplicantIncome = np.sqrt(df_csv_loan_data_of_customer.ApplicantIncome)
            df_csv_loan_data_of_customer.CoapplicantIncome = np.sqrt(df_csv_loan_data_of_customer.CoapplicantIncome)
            df_csv_loan_data_of_customer.LoanAmount = np.sqrt(df_csv_loan_data_of_customer.LoanAmount)
            df_csv_loan_data_of_customer['LoanAmountLog'] = np.log(df_csv_loan_data_of_customer.LoanAmount)
        except Exception:
            print('Error occured in performSquareRootTransformation');
        return df_csv_loan_data_of_customer;
        
        
    def splitDataIntoTrainAndTest(self, df_csv_loan_data_of_customer):
        '''This function returns the splitted data into tran and test'''
        try:
            X = df_csv_loan_data_of_customer.drop(["Loan_Status"], axis=1)
            y = df_csv_loan_data_of_customer["Loan_Status"]
            X, y = SMOTE().fit_resample(X, y)
            scaler = MinMaxScaler()
            scaler.fit(X.values)
            X = scaler.transform(X.values);
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            return X_train, X_test, y_train, y_test, scaler
        except Exception:
            print('Error occured in splitDataIntoTrainAndTest');

        
    def createLogisticRegressionModel(self):
        '''This function create and returns the Logistic Regression Model'''
        logisticRegressionModel = LogisticRegression();
        try:
            logisticRegressionModel = LogisticRegression(solver='saga', max_iter=500, random_state=100)
        except Exception:
            print('Error occured in createLogisticRegressionModel');
        return logisticRegressionModel;
        
        
    def performLogisticRegressionAndPredtict(self, logisticRegressionModel, X_train, y_train, X_test):
        '''This function perfoms the Logistic Regression and predicts'''
        try:
            logisticRegressionModel.fit(X_train, y_train);
            return logisticRegressionModel.predict(X_test)
        except Exception:
            print('Error occured in performLogisticRegressionAndPredtict');
        
        
    def displayCoefficient(self, logisticRegressionModel):
        '''This function displays the Coefficient'''
        try:
            print(logisticRegressionModel.coef_);
        except Exception:
            print('Error occured in displayCoefficient');
    
    def displayIntercept(self, logisticRegressionModel):
        '''This function displays the Intercept'''
        try:
            print(logisticRegressionModel.intercept_);
        except Exception:
            print('Error occured in displayIntercept');
        
    def displayR2Square(self, y_test, y_pred, modelName):
        '''This function displays the R2 Square value of the Model'''
        try:
            r2SquareValue = r2_score(y_test,y_pred)
            print('{} r2_square : {:.2f}%'.format(modelName, (r2SquareValue*100)))
        except Exception:
            print('Error occured in displayR2Square');
        
    
    def displayClassificationReport(self, y_test, y_pred):
        '''This function displays the Classification Report'''
        try:
            print(classification_report(y_test, y_pred));
        except Exception:
            print('Error occured in displayClassificationReport');
        
    def displayConfusionMatric(self, y_test, y_pred):
        '''This function returns the Confusion Matrix'''
        try:
            return confusion_matrix(y_test, y_pred);
        except Exception:
            print('Error occured in displayConfusionMatric');
        
    def displayAccuracySquare(self, y_pred, y_test, modelName):
        '''This function displays the R2 Square value of the Model'''
        try:
            accurScoreValue = accuracy_score(y_pred,y_test)
            print('{}: {:.2f}%'.format(modelName,accurScoreValue*100))
        except Exception:
            print('Error occured in displayAccuracySquare');
         
        
    def displayApplicantsPercentageByEducation(self, df_csv_loan_data_of_customer):
        '''This function displays the Applicants Percentage based on Education'''
        try:
            countOfGraduateApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Education == 'Graduate']);
            countOfNotGraduateApplicants = len(df_csv_loan_data_of_customer[df_csv_loan_data_of_customer.Education == 'Not Graduate']);
            print("The Percentage of Graduate applicants: {:.2f}%".format((countOfGraduateApplicants / (len(df_csv_loan_data_of_customer.Education))*100)));
            print("The Percentage of not Graduate applicants: {:.2f}%".format((countOfNotGraduateApplicants / (len(df_csv_loan_data_of_customer.Education))*100)));
        except Exception:
            print('Error occured in displayApplicantsPercentageByEducation');

    
    def dropUnnecessaryColumns(self, df_csv_loan_data_of_customer):
        '''This function drops the unnecessary Columns from the Data Frame'''
        try:
            df_csv_loan_data_of_customer = df_csv_loan_data_of_customer.drop(['Loan_ID'], axis = 1);
        except Exception:
            print('Error occured in dropUnnecessaryColumns');
        return df_csv_loan_data_of_customer;
        
    
    def createSupportVectorClassifierModel(self):
        '''This function create and returns the Support Vector Classification Model'''
        SVCclassifier = SVC();
        try:
            SVCclassifier = SVC(kernel='rbf', max_iter=500, probability=True)
        except Exception:
            print('Error occured in createSupportVectorClassifierModel');
        return SVCclassifier;

    def performSupportVectorClassifierPredtict(self, SVCclassifier, X_train, y_train, X_test):
        '''This function perfoms the Support Vector Classification and predicts'''
        try:
            SVCclassifier.fit(X_train, y_train)
            return SVCclassifier.predict(X_test)
        except Exception:
            print('Error occured in performSupportVectorClassifierPredtict');
        
    def getLogisticRegressionPredProb(self, logisticRegressionModel, X_train):
        '''This function returns the prediction probability of Logistic Regression Model'''
        try:
            return logisticRegressionModel.predict_proba(X_train)
        except Exception:
            print('Error occured in getLogisticRegressionPredProb');
        
    def getSVCPredProb(self, SVCclassifier, X_train):
        '''This function returns the prediction probability of SupportVectorClassifier'''
        try:
            return SVCclassifier.predict_proba(X_train)
        except Exception:
            print('Error occured in getSVCPredProb');
        
    def compareAndDisplayModelsAccScore(self, X_train, y_train, log_reg_proba, svc_proba):  
        '''This function calculates, compares and Prints the Logistic Regression Model and SupportVectorClassifier Models'''
        try:
            log_reg_auc = round(roc_auc_score(y_train, log_reg_proba[:,1])*100)
            svc_auc = round(roc_auc_score(y_train,svc_proba[:,1])*100)
            print("ROC_AUC_SCORE of models on Train Set\n")
            print("Logistic Regression: ", log_reg_auc)
            print("Support Vector Machine: ", svc_auc)
        except Exception:
            print('Error occured in compareAndDisplayModelsAccScore');
            

        

class LoanPredictionDataPlot():
    
    def drawCountPlotforCategoricalVariables(self, df_csv_loan_data_of_customer):
        '''This function displays a Count Plot of all the Categorical Variables'''
        try:
            categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area' ,'Loan_Status', 'Loan_Amount_Term']
            fig,axes = plt.subplots(4,2,figsize=(12,14))
            for idx, cols in enumerate(categorical_columns):
              row,col = idx//2,idx%2
              sns.countplot(x=cols, data=df_csv_loan_data_of_customer, palette="rocket", ax=axes[row,col])
        except Exception:
            print('Error occured in drawCountPlotforCategoricalVariables');
    
    def drawBarPlotforGenderAndMarried(self, df_csv_loan_data_of_customer):
        '''This function displays a Bar Plot for Gender vs Married'''
        try:
            pd.crosstab(df_csv_loan_data_of_customer.Gender,df_csv_loan_data_of_customer.Married).plot(kind="bar", stacked=True, figsize=(5,5), color=['#f64f59','#12c2e9'])
            plt.title('Gender vs Married')
            plt.xlabel('Gender')
            plt.ylabel('Frequency')
            plt.xticks(rotation=0)
            plt.show()
        except Exception:
            print('Error occured in drawBarPlotforGenderAndMarried');
        
    def drawBarPlotforSelfEmpAndCreditHist(self, df_csv_loan_data_of_customer):
        try:
            '''This function displays a Bar Plot for Self Employeed vs Credit History'''
            pd.crosstab(df_csv_loan_data_of_customer.Self_Employed,df_csv_loan_data_of_customer.Credit_History).plot(kind="bar", stacked=True, figsize=(5,5), color=['#544a7d','#ffd452'])
            plt.title('Self Employed vs Credit History')
            plt.xlabel('Self Employed')
            plt.ylabel('Frequency')
            plt.legend(["Bad Credit", "Good Credit"])
            plt.xticks(rotation=0)
            plt.show()
        except Exception:
            print('Error occured in drawBarPlotforSelfEmpAndCreditHist');
        
    def drawBarPlotforPropertyAreaAndLoanStatus(self, df_csv_loan_data_of_customer):
        '''This function displays a Bar Plot for Property Area vs Loan Status'''
        try:
            pd.crosstab(df_csv_loan_data_of_customer.Property_Area,df_csv_loan_data_of_customer.Loan_Status).plot(kind="bar", stacked=True, figsize=(5,5), color=['#333333','#dd1818'])
            plt.title('Property Area vs Loan Status')
            plt.xlabel('Property Area')
            plt.ylabel('Frequency')
            plt.xticks(rotation=0)
            plt.show()
        except Exception:
            print('Error occured in drawBarPlotforPropertyAreaAndLoanStatus');
            
        
    def drawBoxPlotforLoanStatusAndAppIncome(self, df_csv_loan_data_of_customer):
        '''This function displays a Box Plot for Loan Status vs Applicants Income'''
        try:
            sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df_csv_loan_data_of_customer, palette="mako");
            plt.title('Loan Status vs Applicant Income')
            plt.xlabel('Loan Status')
            plt.ylabel('Applicant Income')
            plt.xticks(rotation=0)
            plt.show()
        except Exception:
            print('Error occured in drawBoxPlotforLoanStatusAndAppIncome');
        
        
    def drawBoxPlotforCoAppIncomeAndLoanStatus(self, df_csv_loan_data_of_customer):
        '''This function displays a Box Plot for Coapplicant Income vs Loan Status'''
        try:
            sns.boxplot(x="CoapplicantIncome", y="Loan_Status", data=df_csv_loan_data_of_customer, palette="rocket");
            plt.title('Coapplicant Income vs Loan Status')
            plt.xlabel('Coapplicant Income')
            plt.ylabel('Loan Status')
            plt.xticks(rotation=0)
            plt.show()
        except Exception:
            print('Error occured in drawBoxPlotforCoAppIncomeAndLoanStatus');
        
        
    def drawBoxPlotforLoanStatusAndLoanAmount(self, df_csv_loan_data_of_customer):
        '''This function displays a Box Plot for Loan Status vs Loan Amount'''
        try:
            sns.boxplot(x="Loan_Status", y="LoanAmount", data=df_csv_loan_data_of_customer, palette="YlOrBr");
            plt.title('Loan Status vs Loan Amount')
            plt.xlabel('Loan Status')
            plt.ylabel('Loan Amount')
            plt.xticks(rotation=0)
            plt.show()
        except Exception:
            print('Error occured in drawBoxPlotforLoanStatusAndLoanAmount');        
            
        
    def drawScatterPlotforAppIncomeAndCoappIncome(self, df_csv_loan_data_of_customer):
        '''This function displays a Scatter Plot for Loan Status vs Loan Amount'''
        try:
            df_csv_loan_data_of_customer.plot(x='ApplicantIncome', y='CoapplicantIncome', kind='scatter', style='o')  
            plt.title('Applicant Income vs CoApplicant Income')  
            plt.xlabel('ApplicantIncome')
            plt.ylabel('CoApplicant Income')  
            plt.show()
            print('Pearson correlation:', df_csv_loan_data_of_customer['ApplicantIncome'].corr(df_csv_loan_data_of_customer['CoapplicantIncome']))
            print('T Test and P value: \n', stats.ttest_ind(df_csv_loan_data_of_customer['ApplicantIncome'], df_csv_loan_data_of_customer['CoapplicantIncome']))
        except Exception:
            print('Error occured in drawScatterPlotforAppIncomeAndCoappIncome'); 

    def drawHistogramPlotforNumericalVariables(self, df_csv_loan_data_of_customer):
        '''This function displays a Histogram Plot of ApplicantIncome vs Count, CoapplicantIncome vs Count, LoanAmount vs Count'''
        try:
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))
            sns.histplot(data=df_csv_loan_data_of_customer, x="ApplicantIncome", kde=True, ax=axs[0], color='green')
            sns.histplot(data=df_csv_loan_data_of_customer, x="CoapplicantIncome", kde=True, ax=axs[1], color='skyblue')
            sns.histplot(data=df_csv_loan_data_of_customer, x="LoanAmount", kde=True, ax=axs[2], color='orange');
        except Exception:
            print('Error occured in drawHistogramPlotforNumericalVariables'); 
        
    def drawHeatMapforApplicantsData(self, df_csv_loan_data_of_customer):
        '''This function displays a Heat Map for Applicants Data'''
        try:
            plt.figure(figsize=(10,7))
            sns.heatmap(df_csv_loan_data_of_customer.corr(), annot=True, cmap='inferno');
        except Exception:
            print('Error occured in drawHeatMapforApplicantsData'); 
        
    def drawROCCureveforModels(self, y_train, log_reg_proba, svc_proba):
        '''This function displays a ROC curve for Logistic Regressiona and SupportVectorClassifier Models'''
        try:
            fpr1, tpr1, thresh1 = roc_curve(y_train, log_reg_proba[:,1],pos_label=1)
            fpr2, tpr2, thresh2 = roc_curve(y_train, svc_proba[:,1],pos_label=1)
            # roc curve for tpr = fpr 
            random_probs = [0 for i in range(len(y_train))]
            p_fpr, p_tpr, _ = roc_curve(y_train, random_probs, pos_label=1)
            plt.style.use('seaborn')
            # plot roc curves
            plt.plot(fpr1, tpr1, linestyle='-',color='orange', label='Logistic Regression')
            plt.plot(fpr2, tpr2, linestyle='-',color='green', label='SVC')
            plt.plot(p_fpr, p_tpr, linestyle='-', color='black')
            # title
            plt.title('ROC curve on Train Set')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')
            plt.legend(loc='best')
            # plt.savefig('ROC',dpi=300)
            plt.show()
        except Exception:
            print('Error occured in drawROCCureveforModels'); 
            
    def drawHeatMapForConfusionMatrix(self, cmatrix):
        '''This function displays the confusion matrix as Heat Map'''
        try:
            plt.figure(figsize=(10,7))
            sns.heatmap(cmatrix, annot=True, cmap='inferno');
            plt.ylabel('Actual label');
            plt.xlabel('Predicted label');
        except Exception:
            print('Error occured in drawHeatMapForConfusionMatrix'); 